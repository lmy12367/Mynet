import torch 
import torch.nn as nn

conv_arch_VGG11 = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
conv_arch_VGG13 = [(2, 64), (2, 128), (2, 256), (2, 512), (2, 512)]
conv_arch_VGG16 = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]

def make_vggblock(in_channel,out_channel,num_convs):
    layers=[]
    
    for _ in range(num_convs):
        layers += [nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1),
                   nn.ReLU()]
        in_channel = out_channel

    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
    

class VGG(nn.Module):
    def __init__(self,con_arh,in_channel,num_class):
        super(VGG,self).__init__()
        self.conv_blk = nn.ModuleList()
        in_ch = in_channel
        
        for num_convs,out_ch in con_arh:
            self.conv_blk.append(
                make_vggblock(in_ch,out_ch,num_convs)
            )
            in_ch = out_ch

        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc=nn.Sequential(
            nn.Linear(out_ch*7*7,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,num_class)
        )
    
    def forward(self,x):
        for blk in self.conv_blk:
            x=blk(x)
        x = self.pool(x)       
        x = torch.flatten(x, 1) 
        x=self.fc(x)
        return x

def vgg(con_arch,in_channel,num_class):
    return VGG(con_arch,in_channel=in_channel,num_class=num_class)

net=vgg(conv_arch_VGG11,in_channel=3,num_class=5)

print(net)  
print(sum(p.numel() for p in net.parameters())) 


