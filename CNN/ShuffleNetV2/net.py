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
    

if __name__=="__main__":
    net=ShuffleNetV2()
    x=torch.randn(1,3,224,224)
    y=net(x)
    print(y.shape)