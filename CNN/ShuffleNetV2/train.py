from random import shuffle
import os, json, time, copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt


from net import ShuffleNetV2      

data_root   = r'./data/flower/flower_split'
model_name  = 'shufflenetv2'         
num_classes = 5

batch_size  = 16
epochs      = 3
lr          = 1e-4
save_dir    = f'./models/{model_name}'
img_dir     = f'./imgs/{model_name}'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_map = {
    'shufflenetv2': ShuffleNetV2
}
net = model_map[model_name](
    stages_repeats=[4, 8, 4],
    stages_out_channels=[24, 116, 232, 464, 1024],  # 1.0× 配置
    num_classes=num_classes
).to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
train_ds = datasets.ImageFolder(os.path.join(data_root, 'train'), transform)
test_ds  = datasets.ImageFolder(os.path.join(data_root, 'test'),  transform)
train_loader = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size, shuffle=False, num_workers=4)

with open(os.path.join(save_dir, 'class_indices.json'), 'w') as f:
    json.dump(train_ds.class_to_idx, f, indent=4)

writer = SummaryWriter(os.path.join(save_dir, 'runs'))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

def train_epoch(epoch):
    net.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    acc = 100. * correct / total
    writer.add_scalar('loss/train', loss_sum / total, epoch)
    writer.add_scalar('acc/train', acc, epoch)
    return acc

@torch.no_grad()
def test_epoch(epoch):
    net.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = net(imgs)
        loss = criterion(outputs, labels)

        loss_sum += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    acc = 100. * correct / total
    writer.add_scalar('loss/test', loss_sum / total, epoch)
    writer.add_scalar('acc/test', acc, epoch)
    return acc

if __name__ == '__main__':
    best_acc, best_wts = 0.0, copy.deepcopy(net.state_dict())
    history = {'train': [], 'test': []}

    print(f'training {model_name} on {device}')
    since = time.time()
    for epoch in range(epochs):
        train_acc = train_epoch(epoch)
        test_acc  = test_epoch(epoch)
        scheduler.step()
        history['train'].append(train_acc)
        history['test'].append(test_acc)
        print(f'epoch {epoch+1:02d}/{epochs}  '
              f'train: {train_acc:.2f}%  test: {test_acc:.2f}%')
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = copy.deepcopy(net.state_dict())
            torch.save(best_wts, os.path.join(save_dir, 'best.pth'))

    torch.save(net.state_dict(), os.path.join(save_dir, 'last.pth'))
    writer.close()
    print(f'best acc: {best_acc:.2f}%  time: {(time.time()-since)/60:.1f}min')

    plt.figure()
    plt.plot(history['train'], label='train')
    plt.plot(history['test'],  label='test')
    plt.xlabel('epoch'); plt.ylabel('accuracy (%)')
    plt.title(f'{model_name} on flower-5')
    plt.legend(); plt.grid()
    plt.savefig(os.path.join(img_dir, 'acc_curve.png'), dpi=300, bbox_inches='tight')