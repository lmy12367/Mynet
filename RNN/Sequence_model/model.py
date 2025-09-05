import torch
from torch import nn
import matplotlib.pyplot as plt

T=1000
time=torch.arange(1,T+1,dtype=torch.float32)
x=torch.sin(0.01*time)+torch.normal(0,0.2,(T,))
print(x.shape)
plt.figure(figsize=(6,3))
plt.plot(time.numpy(), x.numpy())
plt.title("sin box")
plt.xlabel("time")
plt.ylabel("x")
plt.show()

tau=4

features=torch.zeros((T-tau,tau))
print(features.shape)
for i in range(tau):
    features[:,i]=x[i:T-tau+i]

labels=x[tau:].reshape((-1,1))
print("features形状:", features.shape)  
print("labels形状:", labels.shape)    