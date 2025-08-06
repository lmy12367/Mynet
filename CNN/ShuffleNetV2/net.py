from random import shuffle
import re
from turtle import forward
from numpy import reshape
import torch
import torch.nn as nn
from typing import List

class ShuffleNetV2(nn.Module):
    def __init__(self):
        super().__init__()

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




if __name__ == "__main__":
    
    demo_tensor = torch.arange(16).float().view(1, 16, 1, 1)
    print("原始通道顺序:", demo_tensor[0, :, 0, 0])
    shuffled = channel_shuffle(demo_tensor, groups=4)
    print("洗牌后顺序:", shuffled[0, :, 0, 0])