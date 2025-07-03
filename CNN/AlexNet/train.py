from net import MyAlexNet
import matplotlib.pyplot as plt
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F 
import torch.optim as optim
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

batch_size=16
transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307),(0.3081))
])
train_dataset=datasets.MNIST(
    root="D:\\code\\document\\data\\mnist",
    train=True,
    download=False,
    transform=transform
)
train_loader=DataLoader(dataset=train_dataset,
                        shuffle=True,
                        batch_size=batch_size)
test_dataset = datasets.MNIST(
                            root='D:\\code\\document\\data\\\mnist',
                            train=False,
                            download=False,
                            transform=transform)

test_loader = DataLoader(dataset=test_dataset,
                         shuffle=False,
                         batch_size=batch_size,
                         )
model=MyAlexNet()

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    running_loss=0.0
    for batch_index,(inputs,labels) in enumerate(train_loader,0):
        inputs,labels =inputs.to(device),labels.to(device)
        y_pred=model(inputs)
        loss=criterion(y_pred,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if batch_size % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / 100))

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

def save_model(model, filename="D:\code\dp\Review-DP\dpV2\CNN\AlexNet\model.pth"):
    torch.save(model.state_dict(), filename)

if __name__ == '__main__':
    epoch_list = []
    acc_list = []

    for epoch in range(1):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    save_model(model)

    plt.plot(epoch_list, acc_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()