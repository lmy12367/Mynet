import re
from turtle import forward
import torch
import torch.nn as nn
from typing import List

class ShuffleNetV2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x
    
def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """按组把通道重新打乱，便于信息流通"""
    B, C, H, W = x.shape            # 批次、通道、高、宽
    channels_per_group = C // groups
    # 1. reshape -> [B, groups, channels_per_group, H, W]
    x = x.view(B, groups, channels_per_group, H, W)
    # 2. 交换维度 -> [B, channels_per_group, groups, H, W]
    x = x.transpose(1, 2).contiguous()
    # 3. 还原 -> [B, C, H, W]
    x = x.view(B, -1, H, W)
    return x

if __name__=="__main__":
    net=ShuffleNetV2()
    x=torch.randn(1,3,224,224)
    y=net(x)
    print(y.shape)