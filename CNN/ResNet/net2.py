from re import S
from turtle import forward
from networkx import edge_expansion
from numpy import identity
import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=3,
                             stride=stride,
                             padding=1)
        
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu =nn.ReLU()
        self.conv2=nn.Conv2d(in_channels=out_channels,
                             out_channels=out_channels,
                             kernel_size=3,
                             stride=1,
                             padding=1)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.downsample=downsample

    def forward(self,x):
        identity=x
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity=self.downsample(x)
        out = out+identity

        return self.relu(out)
    
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False) 
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels=64
        
        self.conv1=nn.Conv2d(in_channels=3,
                             out_channels=64,
                             kernel_size=7,
                             stride=2,
                             padding=3)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=2,
                                  stride=2,
                                  padding=1)
        
        self.layer1 = self._make_layer(block,64, layers[0])
        self.layer2 = self._make_layer(block,128, layers[1], stride=2)
        self.layer3 = self._make_layer(block,256, layers[2], stride=2)
        self.layer4 = self._make_layer(block,512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels,
                          out_channels=out_channels * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(num_classes=1000):  
    return ResNet(BasicBlock, [2,2,2,2], num_classes)
def resnet34(num_classes=1000):  
    return ResNet(BasicBlock, [3,4,6,3], num_classes)
def resnet50(num_classes=1000):  
    return ResNet(Bottleneck, [3,4,6,3], num_classes)
def resnet101(num_classes=1000): 
    return ResNet(Bottleneck, [3,4,23,3], num_classes)

# model = resnet18(num_classes=5)
# model.eval()

# x = torch.randn(1, 3, 224, 224)

# with torch.no_grad():
#     for name, layer in model.named_children():
#         x = layer(x)
#         if isinstance(layer, torch.nn.AdaptiveAvgPool2d):
#             x = torch.flatten(x, 1)  
#         print(f"{name} ({layer.__class__.__name__}) output shape: {x.shape}")
    

