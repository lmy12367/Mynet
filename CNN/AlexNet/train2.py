from net2 import MyAlexNet  
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F 
import torch.optim as optim
import os
import json


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(
    root="./data/flower/flower_split/train",
    transform=transform
)


test_dataset = datasets.ImageFolder(
    root='./data/flower/flower_split/test',
    transform=transform
)


train_loader = DataLoader(dataset=train_dataset,
                          shuffle=True,
                          batch_size=32)

test_loader = DataLoader(dataset=test_dataset,
                         shuffle=False,
                         batch_size=32)

model = MyAlexNet(num_classes=5) 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
model.to(device)


criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  


def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 10 == 9: 
            print(f'Epoch [{epoch+1}], Batch [{batch_idx+1}], Loss: {running_loss/10:.4f}')
            running_loss = 0.0

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

def save_model(model, filename="./models/flower/flower_classifier.pth"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    torch.save(model.state_dict(), filename)

flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

if __name__ == '__main__':
    acc_history = []
    for epoch in range(10): 
        train(epoch)
        acc = test()
        acc_history.append(acc)
    
    save_model(model)

    plt.plot(range(1, 11), acc_history)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.show()