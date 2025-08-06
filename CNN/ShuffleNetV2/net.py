from random import shuffle
import re
from turtle import forward
from numpy import reshape
import torch
import torch.nn as nn
from typing import List

class ShuffleNetV2(nn.Module):
    def __init__(self,
                stage_repeats:List[int],
                stage_out_channels:List[int],
                num_classes:int=1000):
        super().__init__()

        assert len(stage_repeats)==3,"stages_repeats 必须是 3 个整数"
        assert len(stage_out_channels)==5,"stages_out_channels 必须是 5 个整数"

        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,
                    out_channels=stage_out_channels[0],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
            nn.BatchNorm2d(stage_out_channels[0]),
            nn.ReLU()
        )
        self.maxpool=nn.MaxPool2d(kernel_size=3,
                                stride=2,
                                padding=1)

    def forward(self,x):
        return x
    
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
                nn.ReLU()
            )
        else:
            self.branch1=nn.Identity()

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
            nn.ReLU()
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


if __name__ == "__main__":
    
    demo_tensor = torch.arange(16).float().view(1, 16, 1, 1)
    print("原始通道顺序:", demo_tensor[0, :, 0, 0])
    shuffled = channel_shuffle(demo_tensor, groups=4)
    print("洗牌后顺序:", shuffled[0, :, 0, 0])

    downsample_unit=InvertedResidual(input_channels=24,
                                     output_channels=48,
                                     stride=2)
    x=torch.randn(1,24,56,56)
    y=downsample_unit(x)

    print(y.shape)

    identity_unit=InvertedResidual(input_channels=48,
                                   output_channels=48,
                                   stride=1)
    y2=identity_unit(y)
    print(y2.shape)