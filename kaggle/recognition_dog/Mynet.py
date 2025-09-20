import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def read_train_csv(csv_path):
    df = pd.read_csv(csv_path)
    img_ids = df.iloc[:, 0].values
    breeds = df.iloc[:, 1].values
    return dict(zip(img_ids, breeds))


label_dict = read_train_csv('/kaggle/input/dog-breed-identification/labels.csv')
print(f"Loaded {len(label_dict)} training samples")


def copy_and_organize(data_dir, target_dir, label_dict):
    os.makedirs(target_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(data_dir), desc="Organizing training data"):
        img_id = img_name.split('.')[0]
        breed = label_dict.get(img_id)

        if breed:
            breed_dir = os.path.join(target_dir, breed)
            os.makedirs(breed_dir, exist_ok=True)
            shutil.copy(os.path.join(data_dir, img_name), os.path.join(breed_dir, img_name))


copy_and_organize(
    '/kaggle/input/dog-breed-identification/train',
    '/kaggle/working/train',
    label_dict
)


def organize_test_data(test_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(test_dir), desc="Organizing test data"):
        shutil.copy(
            os.path.join(test_dir, img_name),
            os.path.join(target_dir, img_name)
        )


organize_test_data(
    '/kaggle/input/dog-breed-identification/test',
    '/kaggle/working/test/unknown'
)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(
    '/kaggle/working/train',
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    '/kaggle/working/test',
    transform=test_transform
)

num_classes = len(train_dataset.classes)
print(f"Number of classes: {num_classes}")

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

def create_model(num_classes):
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    return model

model = create_model(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return loss_history

loss_history = train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    num_epochs=10
)

def predict(model, test_loader):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_preds.extend(probs.cpu().numpy())

    return np.array(all_preds)

predictions = predict(model, test_loader)

def create_submission(predictions, test_dir, class_names):
    test_images = sorted(os.listdir(test_dir))
    img_ids = [img.split('.')[0] for img in test_images]

    submission_df = pd.DataFrame(predictions, columns=class_names)
    submission_df.insert(0, 'id', img_ids)

    submission_path = '/kaggle/working/submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    return submission_path


submission_path = create_submission(
    predictions,
    '/kaggle/working/test/unknown',
    train_dataset.classes
)

print("\nSubmission file preview:")
print(pd.read_csv(submission_path).head())
