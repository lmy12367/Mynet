import torch
from torch import nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
        super(BasicBlock,self).__init__()
        self.conv1=nn.Conv2d(input_channels,num_channels,
                             kernel_size=3,padding=1,stride=strides)
        self.conv2=nn.Conv2d(num_channels,num_channels,
                             kernel_size=3,padding=1)
        
        if use_1x1conv:
            self.conv3=nn.Conv2d(input_channels,num_channels,
                                 kernel_size=1,stride=strides)
        else:
            self.conv3=None
        
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self,X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)

        Y += X

        return F.relu(Y)
    
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.b2 = nn.Sequential(*ResNet.resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*ResNet.resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*ResNet.resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*ResNet.resnet_block(256, 512, 2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 10)
    
    def forward(self, X):
        X = self.b1(X)
        X = self.b2(X)
        X = self.b3(X)
        X = self.b4(X)
        X = self.b5(X)
        X = self.avgpool(X)
        X = self.flatten(X)
        X = self.fc(X)
        return X
    
    @staticmethod
    def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
        blk = nn.ModuleList()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(BasicBlock(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
            else:
                blk.append(BasicBlock(num_channels, num_channels))
        return blk
    
net = ResNet()
X = torch.rand(1, 1, 224, 224)
print(net(X).shape)  