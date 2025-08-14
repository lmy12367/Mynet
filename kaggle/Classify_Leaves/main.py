import os, json, pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

ROOT = '/kaggle/input/classify-leaves' if os.path.exists('/kaggle/input') else './data'

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeafDataset(Dataset):
    def __init__(self, csv_file, root_dir=ROOT, transform=None, is_train=True):
        self.df = pd.read_csv(csv_file)
        self.root = root_dir
        self.transform = transform
        self.is_train = is_train
        if self.is_train:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            self.df['label_id'] = le.fit_transform(self.df['label'])
            self.le = le

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.df.iloc[idx]['image'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.is_train:
            return image, self.df.iloc[idx]['label_id']
        return image

train_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_tf = T.Compose([
    T.Resize(int(IMG_SIZE / 0.875)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set = LeafDataset(os.path.join(ROOT, 'train.csv'),
                        transform=train_tf,
                        is_train=True)
num_classes = len(train_set.le.classes_)

train_len, val_len = int(0.8 * len(train_set)), len(train_set) - int(0.8 * len(train_set))
train_ds, val_ds = random_split(train_set, [train_len, val_len],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=2)
test_set = LeafDataset(os.path.join(ROOT, 'test.csv'),
                       transform=test_tf,
                       is_train=False)
test_loader = DataLoader(test_set, BATCH_SIZE,
                         shuffle=False, num_workers=2)


model = resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

def run_epoch(loader, training=True):
    model.train(training)
    total_loss, total_acc, n = 0., 0., 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            pred = model(x)
            loss = criterion(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_acc  += (pred.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, total_acc / n

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)
    print(f'Epoch {epoch:02d}:  '
          f'Train Loss {tr_loss:.4f} Acc {tr_acc:.4f}  |  '
          f'Val Loss {val_loss:.4f} Acc {val_acc:.4f}')

idx2cls = {i: cls for i, cls in enumerate(train_set.le.classes_)}
preds = []
model.eval()
with torch.no_grad():
    for x in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds.extend(logits.argmax(1).cpu().tolist())

sub = pd.read_csv(os.path.join(ROOT, 'test.csv'))
sub['label'] = [idx2cls[p] for p in preds]
sub[['image', 'label']].to_csv('submission.csv', index=False)
print('✅ submission.csv 已生成')