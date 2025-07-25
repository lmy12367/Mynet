import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self,in_c,out_c,**kwargs):
        super(BasicConv2d,self).__init__()
        self.conv=nn.Conv2d(in_c,out_c,**kwargs)
        self.relu=nn.ReLU()

    def forward(self,x):
        return self.relu(self.conv(x))
    

class Inception(nn.Module):
    def __init__(self,in_c,ch1x1,ch3x3red,ch3x3,ch5x5red,ch5x5,pool_proj):
        super(Inception,self).__init__()
        self.branch1=BasicConv2d(in_c,ch1x1,kernel_size=1)

        self.branch2=nn.Sequential(
            BasicConv2d(in_c,ch3x3red,kernel_size=1),
            BasicConv2d(ch3x3red,ch3x3,kernel_size=3,padding=1)
        )

        self.branch3=nn.Sequential(
            BasicConv2d(in_c,ch5x5red,kernel_size=1),
            BasicConv2d(ch5x5red,ch5x5,kernel_size=5,padding=2)
        )

        self.branch4=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            BasicConv2d(in_c,pool_proj,kernel_size=1)
        )

    def forward(self,x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ],1)

class InceptionAux(nn.Module):
    def __init__(self,in_channels, num_classes):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5,stride=3)
        self.conv=BasicConv2d(in_channels,128,kernel_size=1)

        self.fc1=nn.Linear(128*4*4,1024)
        self.fc2=nn.Linear(1024,num_classes)

    def forward(self,x):
        x=self.avgpool(x)
        x=self.conv(x)
        x=torch.flatten(x,1)
        
        x=F.dropout(x,0.5,self.training)
        x=F.relu(self.fc1(x))
        x=F.dropout(x,0.5,self.training)
        x=self.fc2(x)

        return x
    
class GoogLenet(nn.Module):
    def __init__(self,num_class=5,autx_logits=True):
        super().__init__()
        self.autx_logits=autx_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
    
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3   = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4   = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        
        if self.autx_logits:
            self.aux1 = InceptionAux(512, num_class)  
            self.aux2 = InceptionAux(528, num_class)  
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        if self.training and self.autx_logits:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.training and self.autx_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.autx_logits:
            return x, aux2, aux1
        return x