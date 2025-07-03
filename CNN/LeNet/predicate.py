import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from net import MyLenet

def load_model(device):
    model = MyLenet()
    model.load_state_dict(torch.load('D:\\code\\dp\\Review-DP\\dpV2\\CNN\\LeNEt\\model.pth'))
    model.to(device)
    model.eval()
    return model

def predict(model, device, test_loader):
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
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)

    test_dataset = datasets.MNIST(
        root="D:\\code\\document\\data\\\mnist",
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081))
        ])
    )
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=32)

    predict(model, device, test_loader)