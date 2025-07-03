import torch
from torch import nn

class MyLenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2)
        self.pooling1=nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.pooling2=nn.AvgPool2d(kernel_size=2,stride=2)

        self.linear1=nn.Linear(16*5*5,120)
        self.linear2=nn.Linear(120,84)
        self.linear3=nn.Linear(84,10)
    def forward(self,x):
        x=torch.relu(self.conv1(x))
        x=self.pooling1(x)
        x=torch.relu(self.conv2(x))
        x=self.pooling2(x)
        x = x.view(x.size(0), -1)
        x=torch.relu(self.linear1(x))
        x=torch.relu(self.linear2(x))
        x=self.linear3(x)
        return x
