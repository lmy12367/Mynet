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



if __name__ == "__main__":
    x = torch.randn(1, 8, 4, 4)
    y = channel_shuffle(x, 2)      # 2 组
    print("洗牌前:", x[0, :, 0, 0])
    print("洗牌后:", y[0, :, 0, 0])