import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features.astype('float32')

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
dataset = TensorDataset(train_features, train_labels)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory_device="cuda")

