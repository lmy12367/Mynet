import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split

train_df=pd.read_csv("./data/train.csv")
print(train_df.head())
print(len(train_df))

print("类别数",train_df['label'].nunique())
le=LabelEncoder()
train_df['label_id']=le.fit_transform(train_df['label'])

with open('./data/label2idx.json', 'w') as f:
    json.dump({cls: idx for idx, cls in enumerate(le.classes_)},f, indent=2)
print('映射已保存到 ./data/label2idx.json')

class LeafDataset(Dataset):
    def __init__(self, csv_file, root_dir='./data', transform=None, is_train=True):
        self.df = pd.read_csv(csv_file)
        self.root = root_dir
        self.transform = transform
        self.is_train = is_train

        # 👇 训练 csv 现场加 label_id
        if self.is_train:
            le = LabelEncoder()
            self.df['label_id'] = le.fit_transform(self.df['label'])
            self.le = le        # 可选：反向映射用

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.df.iloc[idx]['image'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.is_train:
            label = self.df.iloc[idx]['label_id']   # 现在一定有了
            return image, label
        else:
            return image

ds = LeafDataset('./data/train.csv')
img, lbl = ds[0]
print(type(img), img.size, lbl)

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(MEAN, STD)
])

test_tf = T.Compose([
    T.Resize(int(IMG_SIZE / 0.875)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(MEAN, STD)
])

ds_tf = LeafDataset('./data/train.csv')
ds_tf.transform = train_tf
tensor, lbl = ds_tf[0]
print(tensor.shape, lbl)

train_set = LeafDataset('./data/train.csv')
train_set.transform = train_tf

train_len = int(0.7 * len(train_set))
val_len   = len(train_set) - train_len
train_ds, val_ds = random_split(train_set, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,  num_workers=4)

print('训练 batch 数:', len(train_loader))
print('验证 batch 数:', len(val_loader))

test_set = LeafDataset('./data/test.csv', is_train=False)
test_set.transform = test_tf
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

print('测试 batch 数:', len(test_loader))