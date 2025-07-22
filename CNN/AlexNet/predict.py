import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from net2 import MyAlexNet         

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

    img_path = "./dp/Mynet/CNN/AlexNet/test.jpg"
    assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
    img = Image.open(img_path).convert('RGB')   
    plt.imshow(img)

   
    img = data_transform(img)          
    img = torch.unsqueeze(img, dim=0)  


    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, 'r') as f:
        class_indict = json.load(f)


    model = MyAlexNet(num_classes=5).to(device)


    weights_path = "./models/flower/flower_classifier.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' does not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        output = model(img.to(device))          
        output = torch.squeeze(output)        
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).item()


    print_res = "class: {}   prob: {:.3f}".format(
        class_indict[str(predict_cla)], predict[predict_cla].item())
    plt.title(print_res)

    for idx, prob in enumerate(predict):
        print("class: {:10}   prob: {:.3f}".format(class_indict[str(idx)], prob.item()))

    plt.show()


if __name__ == '__main__':
    main()