import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import time
from net import ResNet

batch_size = 128

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root="D:\\code\\document\\data\\mnist",
    train=True,
    download=False,
    transform=transform_train
)
train_dataset = Subset(train_dataset, range(2048))  
train_loader = DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=batch_size,
    num_workers=4 
)


test_dataset = torchvision.datasets.MNIST(
    root="D:\\code\\document\\data\\mnist",
    train=False,
    download=False,
    transform=transform_test
)
test_dataset = Subset(test_dataset, range(512))  
test_loader = DataLoader(
    dataset=test_dataset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=4
)

model = ResNet()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)


criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5, weight_decay=5e-4)  


def train(epoch):
    model.train()
    running_loss = 0.0
    start_time = time.time()  
    for batch_index, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_index % 100 == 99:
            avg_loss = running_loss / 100
            print(f'Epoch [{epoch+1}], Batch [{batch_index+1}], Loss: {avg_loss:.4f}')
            running_loss = 0.0

    end_time = time.time()  
    print(f'Epoch [{epoch+1}] completed in {end_time - start_time:.2f} seconds')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss / len(test_loader):.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

def save_model(model, filename="D:\\code\dp\\Review-DP\\dpV2\\Data_augmentation\\model.pth"):
    try:
        torch.save(model.state_dict(), filename)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Failed to save model: {e}")


if __name__ == '__main__':
    epoch_list = []
    acc_list = []
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")
    print(f"显存使用情况: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB (初始)")

    for epoch in range(8):  
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    save_model(model)

    plt.plot(epoch_list, acc_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()