import torch
import torch.nn as nn
import  torch.nn.functional as F

class ClassTitanic(nn.Module):
    def __init__(self,input_features):
        super(ClassTitanic,self).__init__()
        self.linear1=nn.Linear(input_features,64)
        self.linear2=nn.Linear(64,64)
        self.linear3=nn.Linear(64,10)
        self.linear4=nn.Linear(10,1)

        self.dropout=nn.Dropout(0.2)

        self.batchnorm1=nn.BatchNorm1d(64)
        self.batchnorm2 =nn.BatchNorm1d(64)
        self.batchnorm3=nn.BatchNorm1d(10)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.linear1(x)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm2(self.linear2(x)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm3(self.linear3(x)))
        x = self.dropout(x)
        x = self.linear4(x)
        return torch.sigmoid(x)