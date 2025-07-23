from  net2 import vgg,conv_arch_VGG11
import matplotlib.pyplot as plt
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import os
import json

batch_size=16
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset=datasets.ImageFolder(
    root="./data/flower/flower_split/train",
    transform=transform
)

train_loader=DataLoader(dataset=train_dataset,
                        shuffle=True,
                        batch_size=batch_size)

test_dataset=datasets.ImageFolder(
    root='./data/flower/flower_split/test',
    transform=transform
)

test_loader=DataLoader(dataset=test_dataset,
                       shuffle=False,
                       batch_size=batch_size)
model=vgg(conv_arch_VGG11,in_channel=3,num_class=5)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion=torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train(epoch):
    model.train()
    running_loss=0.0

    for batch_idx,(input,labels) in enumerate(train_loader,0):

        input,labels=input.to(device),labels.to(device)
        y_pred=model(input)
        loss=criterion(y_pred,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx %10 ==9:
            avg_loss =running_loss/100
            print(f'Epoch [{epoch+1}], Batch [{batch_idx+1}], Loss: {avg_loss:.4f}')
            running_loss = 0.0
            
            
            mem_used = torch.cuda.memory_allocated(device)/1024**3
            print(f'Epoch [{epoch+1}], Batch [{batch_idx+1}], '
                  f'Loss: {avg_loss:.4f}, Mem: {mem_used:.2f} GB')
            
            
            if mem_used > 7.5:  
                print("警告：显存使用接近极限！")
                torch.cuda.empty_cache()

def test():
    model.eval()
    test_loss=0.0
    correct=0
    with torch.no_grad():
        for input,labels in test_loader:
            input,labels=input.to(device),labels.to(device)
            outputs=model(input)
            loss=criterion(outputs,labels)
            test_loss += loss.item()
            _,predicted = torch.max(outputs.data,1)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss / len(test_loader):.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


def save_model(model,filename="./models/VGG/flower_model.pth"):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)   
        torch.save(model.state_dict(), filename)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Failed to save model: {e}")

flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

if __name__ == '__main__':
    epoch_list = []
    acc_list = []

    for epoch in range(15):
        train(epoch)
        acc = test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    save_model(model)


    save_dir_img = "./dp/dpV2/CNN/VGG/imgs"
    os.makedirs(save_dir_img, exist_ok=True)

    plt.figure()
    plt.plot(epoch_list, acc_list, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Flower-5 Accuracy')
    plt.grid()
    plt.savefig(os.path.join(save_dir_img, "acc_curve.png"), dpi=300, bbox_inches='tight')
    plt.show()