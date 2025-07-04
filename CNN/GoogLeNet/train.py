from net import GoogLeNet
import matplotlib.pyplot as plt
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F 
import torch.optim as optim


batch_size=4
transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
])

train_dataset=datasets.MNIST(
    root="D:\\code\\document\\data\\mnist",
    train=True,
    download=False,
    transform=transform
)
train_dataset = torch.utils.data.Subset(train_dataset, range(512))
train_loader=DataLoader(dataset=train_dataset,
                        shuffle=True,
                        batch_size=batch_size)
test_dataset = datasets.MNIST(
                            root='D:\\code\\document\\data\\\mnist',
                            train=False,
                            download=False,
                            transform=transform)

test_dataset = torch.utils.data.Subset(test_dataset, range(128))

test_loader = DataLoader(dataset=test_dataset,
                         shuffle=False,
                         batch_size=batch_size,
                         )

model = GoogLeNet()

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数总数: {total_params:,}")
print(f"显存使用情况: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB (初始)")

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer=optim.SGD(model.parameters(),lr=0.1,momentum=0.5)

def train(epoch):
    model.train()
    running_loss=0.0
    for batch_index,(inputs,labels) in enumerate(train_loader,0):
        
        inputs,labels =inputs.to(device),labels.to(device)
        y_pred=model(inputs)
        loss=criterion(y_pred,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if batch_index % 100 == 99:
            avg_loss = running_loss / 100
            print(f'Epoch [{epoch+1}], Batch [{batch_index+1}], Loss: {avg_loss:.4f}')
            running_loss = 0.0

            mem_used = torch.cuda.memory_allocated(device)/1024**3
            print(f'Epoch [{epoch+1}], Batch [{batch_index+1}], '
                  f'Loss: {avg_loss:.4f}, Mem: {mem_used:.2f} GB')
            
            # 显存安全检查
            if mem_used > 7.5:  # 7.5GB警告
                print("警告：显存使用接近极限！")
                torch.cuda.empty_cache()

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

def save_model(model, filename="D:\code\dp\Review-DP\dpV2\CNN\VGG\model.pth"):
    try:
        torch.save(model.state_dict(), filename)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Failed to save model: {e}")

if __name__ == '__main__':
    epoch_list = []
    acc_list = []

    for epoch in range(2):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    save_model(model)

    plt.plot(epoch_list, acc_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()