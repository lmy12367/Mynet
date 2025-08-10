import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self,in_channel,hidden1=200,hidden2=100,out_channel=1):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_channel,hidden1),
            nn.ReLU(),
            nn.Linear(hidden1,hidden2),
            nn.ReLU(),
            nn.Linear(hidden2,out_channel)
        )

    def forward(self,x):
        return self.net(x)

# if __name__=="__main__":
#     net=MyNet(331)
#     print(net)