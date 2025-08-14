import torch, json, pandas as pd
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from data_preprocess import LeafDataset
from torchvision import transforms as T

BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-4
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = test_tf = T.Compose([
    T.Resize(int(IMG_SIZE/0.875)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def main():
    root = './data'

    train_df_small = pd.read_csv(root + '/train.csv').head(512)
    train_df_small.to_csv(root + '/train_small.csv', index=False)

    train_full = LeafDataset(root + '/train.csv', transform=train_tf, is_train=True)
    num_classes = len(train_full.le.classes_)
    train_len = int(0.7 * len(train_full))
    val_len   = len(train_full) - train_len
    train_ds, val_ds = random_split(train_full, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(
        LeafDataset(root + '/test.csv', transform=test_tf, is_train=False),
        BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch} loss {loss.item():.4f}')

    torch.save(model.state_dict(), 'best.pth')
    print('训练完成，权重已保存为 best.pth')

if __name__ == '__main__':
    main()