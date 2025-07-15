import torch
import torchvision.transforms as transforms
from PIL import Image
from net import MyLenet

def main():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),  
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))    
    ])

    classes = [str(i) for i in range(10)]

    net = MyLenet()
    net.load_state_dict(torch.load(r'./models/model.pth',
                                   map_location='cpu'))
    net.eval()

    img_path = r'./mnist_sample.png'
    im = Image.open(img_path).convert('L')  
    im = transform(im).unsqueeze(0)         

    with torch.no_grad():
        outputs = net(im)
        pred = int(torch.argmax(outputs, dim=1))
    print('Predicted digit:', classes[pred])

if __name__ == '__main__':
    main()