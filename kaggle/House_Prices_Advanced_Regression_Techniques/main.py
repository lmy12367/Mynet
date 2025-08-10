import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class MyNet(nn.Module):
    def __init__(self, in_channel, hidden1=200, hidden2=100, out_channel=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channel, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, out_channel)
        )

    def forward(self, x):
        return self.net(x)

def get_data(batch_size=32):
    base_dir = '/kaggle/input'
    train_path = None
    test_path = None

    for root, _, files in os.walk(base_dir):
        if 'train.csv' in files:
            train_path = os.path.join(root, 'train.csv')
        if 'test.csv' in files:
            test_path = os.path.join(root, 'test.csv')

    if train_path is None or test_path is None:
        raise FileNotFoundError("未找到 train.csv 或 test.csv，请确认数据集已正确挂载。")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    all_features = pd.concat((train_df.iloc[:, 1:-1], test_df.iloc[:, 1:]))

    numeric_cols = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_cols] = all_features[numeric_cols].apply(
        lambda x: (x - x.mean()) / (x.std())
    ).fillna(0)

    all_features = pd.get_dummies(all_features, dummy_na=True).astype(np.float32)

    n_train = len(train_df)
    train_X = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_X  = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    train_y = torch.tensor(train_df['SalePrice'].values.reshape(-1, 1), dtype=torch.float32)

    dataset = TensorDataset(train_X, train_y)
    loader  = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
    return train_df, test_df, train_X, test_X, train_y, loader

def train(epochs=200, lr=1e-2, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_df, test_df, train_X, test_X, train_y, train_loader = get_data(batch_size)
    in_channel = train_X.shape[1]

    model = MyNet(in_channel).to(device)
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_list = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_X)

        epoch_loss /= len(train_loader)
        loss_list.append(epoch_loss)
        if epoch % 20 == 0 or epoch == 1:
            print(f"[train] Epoch {epoch:3d} / {epochs} | loss = {epoch_loss:.4f}")

    plt.plot(range(1, epochs + 1), loss_list)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig("/kaggle/working/training_loss_last.png")
    plt.close()

    # 保存模型权重
    torch.save(model.state_dict(), "/kaggle/working/model_last.pth")
    print("[train] 模型已保存到 /kaggle/working/model_last.pth")
    return model, test_df, test_X

def inference(model, test_df, test_X):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        preds = model(test_X.to(device)).cpu().numpy().squeeze()

    submission = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": preds
    })
    submission.to_csv("/kaggle/working/submission.csv", index=False)
    print("[main] /kaggle/working/submission.csv 已生成！")

def main():
    model, test_df, test_X = train()
    inference(model, test_df, test_X)

if __name__ == "__main__":
    main()