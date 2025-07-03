import torch
from torch import nn

class VGG(nn.Module):
    def __init__(self, conv_arch):
        super(VGG, self).__init__()
        self.conv_blks = nn.ModuleList()  
        in_channels = 1
        for (num_convs, out_channels) in conv_arch:
            conv_blk = []
            for _ in range(num_convs):
                conv_blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                conv_blk.append(nn.ReLU())
                in_channels = out_channels
            conv_blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv_blks.append(nn.Sequential(*conv_blk))  
        
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(out_channels * 7 * 7, 4096),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        for conv_blk in self.conv_blks:
            x = conv_blk(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def vgg(conv_arch):
    return VGG(conv_arch)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
net = vgg(conv_arch)

print(net)  
print(sum(p.numel() for p in net.parameters())) 

