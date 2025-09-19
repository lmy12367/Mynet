import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class TitanicDataset(Dataset):
    def __init__(self, train_path='./train.csv', test_path='./test.csv', is_train=True, scaler=None):
        super(TitanicDataset, self).__init__()

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        if is_train:
            self.features, self.labels, self.test_ids = self.dataprocess(train_data, test_data, is_train=True)
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.features, self.test_ids = self.dataprocess(train_data, test_data, is_train=False)
            if scaler is not None:
                self.scaler = scaler
                self.features = self.scaler.transform(self.features)
            else:
                raise ValueError("测试模式下必须提供训练好的scaler")

        self.is_train = is_train

    def dataprocess(self, train_data, test_data, is_train=True):
        if is_train:
            df = train_data.copy()
            test_ids = None
        else:
            df = test_data.copy()
            test_ids = df['PassengerId'].values

        cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        if 'Name' in df.columns:
            df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                               'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            df['Title'] = df['Title'].replace('Mlle', 'Miss')
            df['Title'] = df['Title'].replace('Ms', 'Miss')
            df['Title'] = df['Title'].replace('Mme', 'Mrs')
            title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
            df['Title'] = df['Title'].map(title_mapping).fillna(0)

        sex_map = {'male': 0, 'female': 1}
        df['Sex'] = df['Sex'].map(sex_map)

        emb_map = {'S': 0, 'C': 1, 'Q': 2}
        df['Embarked'] = df['Embarked'].map(emb_map)

        if 'Name' in df.columns:
            df = df.drop(columns=['Name'])

        if is_train:
            labels = df['Survived'].values
            features = df.drop(columns=['Survived']).values
            return features, labels, test_ids
        else:
            features = df.values
            return features, test_ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.is_train:
            features = torch.tensor(self.features[idx], dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return features, label
        else:
            features = torch.tensor(self.features[idx], dtype=torch.float32)
            test_id = self.test_ids[idx]
            return features, test_id

def collate_fn_test(batch):
    features = torch.stack([item[0] for item in batch])
    ids = [item[1] for item in batch]
    return features, ids

traindataset = TitanicDataset(train_path='./train.csv', is_train=True)
trainloader = DataLoader(traindataset, batch_size=32, shuffle=True)
testdataset = TitanicDataset(test_path='./test.csv', is_train=False, scaler=traindataset.scaler)
testloader = DataLoader(testdataset, batch_size=32, shuffle=False, collate_fn=collate_fn_test)
for features, labels in trainloader:
    print(f"训练批次 - 特征形状: {features.shape}, 标签形状: {labels.shape}")

for features, ids in testloader:
    print(f"测试批次 - 特征形状: {features.shape}, ID数量: {len(ids)}")
