import torch
import torch.nn as nn
from typing import List

class ShuffleNetV2(nn.Module):
    def __init__(self,
                stages_repeats:List[int],
                stages_out_channels:List[int],
                num_classes:int=1000):
        super().__init__()

        assert len(stages_repeats)==3,"stages_repeats 必须是 3 个整数"
        assert len(stages_out_channels)==5,"stages_out_channels 必须是 5 个整数"

        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,
                    out_channels=stages_out_channels[0],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
            nn.BatchNorm2d(stages_out_channels[0]),
            nn.ReLU()
        )
        self.maxpool=nn.MaxPool2d(kernel_size=3,
                                stride=2,
                                padding=1)
        current_channels=stages_out_channels[0]
        stage_names = ["stage2", "stage3", "stage4"]
        for stage_name,repeats,output_channels in zip(
            stage_names,stages_repeats,stages_out_channels[1:]):
            stage_blocks=[InvertedResidual(current_channels,
                                           output_channels,2)]
            stage_blocks+=[InvertedResidual(output_channels, 
                                            output_channels,1)
                                            for _ in range(repeats-1)] 
            
            setattr(self,stage_name,nn.Sequential(*stage_blocks))
            current_channels=output_channels

        self.conv5=nn.Sequential(
            nn.Conv2d(current_channels,
                      stages_out_channels[-1],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(stages_out_channels[-1]),
            nn.ReLU())
        self.global_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(stages_out_channels[-1],num_classes)


    def forward(self,input_image:torch.Tensor)->torch.Tensor:
        feature_map=self.conv1(input_image)
        feature_map=self.maxpool(feature_map)
        feature_map=self.stage2(feature_map)
        feature_map=self.stage3(feature_map)
        feature_map=self.stage4(feature_map)
        feature_map=self.conv5(feature_map)
        feature_map=self.global_pool(feature_map).flatten(1)
        return self.fc(feature_map)
        
    
def channel_shuffle(input_tensor:torch.Tensor,groups:int)->torch.Tensor:
    batch_size,num_channels,height,width=input_tensor.shape
    channels_per_group=num_channels//groups

    reshaped_tensor=input_tensor.view(
        batch_size,
        groups,
        channels_per_group,
        height,
        width
    )

    transposed_tensor=reshaped_tensor.transpose(1,2).contiguous()

    shuffled_tensor=transposed_tensor.view(
        batch_size,
        num_channels,
        height,
        width
    )

    return shuffled_tensor

class InvertedResidual(nn.Module):
    def __init__(self,
                input_channels:int,
                output_channels:int,
                stride:int):
        super().__init__()
        self.stride=stride
        assert stride in [1,2]
        branch_features=output_channels//2

        assert(stride!=1)or(input_channels==branch_features*2)

        if stride==2:
            self.branch1=nn.Sequential(
                nn.Conv2d(input_channels,
                        input_channels,
                        kernel_size=3, 
                        stride=stride,
                        padding=1,
                        groups=input_channels,
                        bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(),
                nn.Conv2d(input_channels,
                        branch_features,
                        kernel_size=1, 
                        stride=1,
                        padding=0,
                        bias=False),
                        nn.BatchNorm2d(branch_features)
    )

    

        else:
            self.branch1 = nn.Identity()



        self.branch2=nn.Sequential(
            nn.Conv2d(input_channels if stride>1 else branch_features,
                      branch_features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(),
            nn.Conv2d(branch_features,
                      branch_features,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=branch_features,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features,
                      branch_features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
        )
    
    def forward(self,input_feature_map:torch.Tensor):
        if self.stride==1:
            branch_left,branch_right=input_feature_map.chunk(2,dim=1)
            output_feature_map=torch.cat(
                (branch_left,self.branch2(branch_right)),dim=1
            )
        else:
            output_feature_map=torch.cat(
                (self.branch1(input_feature_map),
                 self.branch2(input_feature_map)),dim=1
            )
    
        return channel_shuffle(output_feature_map,2)


# if __name__ == "__main__":
#     net = ShuffleNetV2(
#         stages_repeats=[4, 8, 4],
#         stages_out_channels=[24, 116, 232, 464, 1024],  # ← 原来写成了 82
#         num_classes=1000
#     )
#     x = torch.randn(1, 3, 224, 224)
#     y = net(x)
#     print("输出形状:", y.shape)  # [1, 1000]