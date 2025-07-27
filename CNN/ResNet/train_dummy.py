import torch
from torch.utils.tensorboard import SummaryWriter
from net import ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ResNet(num_classes=5).to(device)

writer = SummaryWriter('./runs_dummy')


dummy_cpu = torch.randn(3, 3, 224, 224)
writer.add_graph(net.cpu(), dummy_cpu)
net.to(device)         
del dummy_cpu

for step in range(100):
    loss = torch.rand(1).item()
    writer.add_scalar('loss/train', loss, step)

writer.close()
print("Dummy 图已写入 ./runs_dummy，执行：\n"
      "tensorboard --logdir=./runs_dummy")