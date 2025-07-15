from netv2 import LeNet       
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001                       
SAVE_PATH = './models/lenet_cifar10_small.pth'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

full_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=False, transform=transform)
full_test  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=False, transform=transform)

train_subset = Subset(full_train, indices=range(1000))
test_subset  = Subset(full_test,  indices=range(500))

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_subset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

model = LeNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:   
            print(f'Epoch {epoch}[{batch_idx}/{len(train_loader)}] '
                  f'loss: {running_loss/10:.3f}')
            running_loss = 0.0

def test():
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    print(f'Test: Avg loss={avg_loss:.4f}, '
          f'Accuracy={correct}/{total} ({accuracy:.2f}%)')
    return accuracy

if __name__ == '__main__':
    epoch_list, acc_list = [], []

    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    torch.save(model.state_dict(), SAVE_PATH)
    print(f'Model saved to {SAVE_PATH}')

    plt.plot(epoch_list, acc_list, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('LeNet on CIFAR-10 (1000/500)')
    plt.grid()
    plt.show()