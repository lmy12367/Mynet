import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import  torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os

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

class ClassTitanic(nn.Module):
    def __init__(self,input_features):
        super(ClassTitanic,self).__init__()
        self.linear1=nn.Linear(input_features,64)
        self.linear2=nn.Linear(64,64)
        self.linear3=nn.Linear(64,10)
        self.linear4=nn.Linear(10,1)

        self.dropout=nn.Dropout(0.2)

        self.batchnorm1=nn.BatchNorm1d(64)
        self.batchnorm2 =nn.BatchNorm1d(64)
        self.batchnorm3=nn.BatchNorm1d(10)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.linear1(x)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm2(self.linear2(x)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm3(self.linear3(x)))
        x = self.dropout(x)
        x = self.linear4(x)
        return torch.sigmoid(x)


def train_model(model, train_loader, criterion, optimizer, num_epochs=25, device='cuda'):
    model = model.to(device)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f'Epoch{epoch + 1}/{num_epochs}')
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'models/best_model.pth')
            print("Best model saved as models/best_model.pth")

        torch.save(model.state_dict(), 'models/final_model.pth')
        print("Final model saved as models/final_model.pth")

    print('Training completes')
    return model


def predict_and_save(model, test_loader, device='cuda'):
    model = model.to(device)
    model.eval()

    predictions = []
    ids = []

    with torch.no_grad():
        for inputs, test_ids in tqdm(test_loader, desc='Predicting'):
            inputs = inputs.to(device)
            outputs = model(inputs)

            preds = (outputs > 0.5).float().cpu().numpy().flatten()
            predictions.extend(preds)
            ids.extend(test_ids)

    predictions = np.array(predictions).astype(int)

    submission = pd.DataFrame({
        'PassengerId': ids,
        'Survived': predictions
    })

    submission = submission.sort_values('PassengerId')

    submission.to_csv('submission.csv', index=False)
    print("Submission saved as submission.csv")

    return submission


def main():
    os.makedirs('models', exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    batch_size = 32
    num_epochs = 200
    learning_rate = 0.001
    train_dataset = TitanicDataset(train_path='./train.csv', test_path='./test.csv', is_train=True)
    test_dataset = TitanicDataset(train_path='./train.csv', test_path='./test.csv', is_train=False,
                                  scaler=train_dataset.scaler)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_test)
    input_features = train_dataset.features.shape[1]
    print(f'Number of features: {input_features}')
    model = ClassTitanic(input_features)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs, device=device)
    model.load_state_dict(torch.load('models/best_model.pth'))
    submission = predict_and_save(model, test_loader, device=device)
    print("\nSubmission file preview:")
    print(submission.head())


if __name__ == '__main__':
    main()



