from net2 import GoogLenet     
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import os
import json

batch_size = 16
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   #
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(
    root="./data/flower/flower_split/train",
    transform=transform
)
train_loader = DataLoader(dataset=train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.ImageFolder(
    root='./data/flower/flower_split/test',
    transform=transform
)
test_loader = DataLoader(dataset=test_dataset,
                         shuffle=False,
                         batch_size=batch_size)

model = GoogLenet(num_class=5, autx_logits=True)   

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train(epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)


        outputs = model(inputs)
        if isinstance(outputs, tuple):  
            logits, aux2, aux1 = outputs
            loss_main = criterion(logits, labels)
            loss_aux1 = criterion(aux1, labels)
            loss_aux2 = criterion(aux2, labels)
        
            loss = loss_main + 0.3 * (loss_aux1 + loss_aux2)
        else:                       
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            avg_loss = running_loss / 10
            print(f'Epoch [{epoch+1}], Batch [{batch_idx}], Loss: {avg_loss:.4f}')
            running_loss = 0.0

            mem_used = torch.cuda.memory_allocated(device) / 1024**3
            print(f'Mem: {mem_used:.2f} GB')
            if mem_used > 7.5:
                print("警告：显存使用接近极限！")
                torch.cuda.empty_cache()


def test():
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
   
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / len(test_loader.dataset)
    print(f'Test: loss={test_loss/len(test_loader):.4f}, '
          f'acc={accuracy:.2f}% ({correct}/{len(test_loader.dataset)})')
    return accuracy


def save_model(model, filename="./models/GoogLeNet/flower_model.pth"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

flower_list = train_dataset.class_to_idx
cla_dict = dict((v, k) for k, v in flower_list.items())
with open('class_indices.json', 'w') as f:
    json.dump(cla_dict, f, indent=4)

if __name__ == '__main__':
    epoch_list, acc_list = [], []

    for epoch in range(15):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    save_model(model)

    save_dir_img = "./dp/dpV2/CNN/GoogLeNet/imgs"
    os.makedirs(save_dir_img, exist_ok=True)
    plt.figure()
    plt.plot(epoch_list, acc_list, marker='o')
    plt.xlabel('epoch'); plt.ylabel('accuracy')
    plt.title('Flower-5 Accuracy (GoogLeNet)')
    plt.grid()
    plt.savefig(os.path.join(save_dir_img, "acc_curve.png"), dpi=300, bbox_inches='tight')
    plt.show()