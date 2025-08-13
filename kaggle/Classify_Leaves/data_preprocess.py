import json

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tensorflow.compiler.tf2xla.python.xla import shift_left
from torch.utils.data import Dataset

train_df=pd.read_csv("./data/train.csv")
test_df=pd.read_csv("./data/test.csv")

print(train_df.head())
print(len(train_df))
print((len(test_df)))
print(train_df['label'].nunique())

le = LabelEncoder()
train_df['label_id'] = le.fit_transform(train_df['label'])
num_classes=len(le.classes_)
print(num_classes)

with open("./data/label2idx.json",'w') as f:
    json.dump({l:i for i,l in enumerate(le.classes_)},f)

class LeafDataset(Dataset):
    def __init__(self,csv_file,root_dir='./data',
                 transform=None,is_train=None):
        self.df=pd.read_csv(csv_file)
        self.root=root_dir
        self.transform=transform
        self.is_train=is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path=os.path.join(self.root,
                              self.df.iloc[idx]["image"])
        image=Image.open(img_path).convert("RGB")

        if self.transform:
            image=self.transform(image)

        if self.is_train:
            label = self.df.iloc[idx]['label_idx']
            return image, label
        else:
            return image


