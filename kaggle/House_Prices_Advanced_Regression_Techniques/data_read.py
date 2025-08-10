import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data():
    train_df=pd.read_csv("./data/train.csv")
    test_df=pd.read_csv('./data/test.csv')

    print(f"{train_df.shape}")
    print(f'{test_df.shape}')

    all_features=pd.concat(
        (train_df.iloc[:,1:-1],test_df.iloc[:,1:])
    )

    print(f"[load] 合并后 all_features.shape = {all_features.shape}")
    return train_df, test_df, all_features

def preprocess(train_df,test_df,all_features):
    numeroc_clos=all_features.dtypes[all_features.dtypes!='object'].index
    print({len(numeroc_clos)})

    all_features[numeroc_clos]=all_features[numeroc_clos].apply(
        lambda x:(x-x.mean())/(x.std())
    ).fillna(0)

    all_features=pd.get_dummies(all_features,dummy_na=True)
    all_features = all_features.astype(np.float32)
    print(f'{all_features.shape}')

    n_train=len(train_df)
    train_X=torch.tensor(all_features[:n_train].values,dtype=torch.float32)
    test_X=torch.tensor(all_features[n_train:].values,dtype=torch.float32)
    train_y=torch.tensor(train_df['SalePrice'].values.reshape(-1,1),dtype=torch.float32)

    print(f"[preprocess] 最终 train_X.shape = {train_X.shape}")
    print(f"[preprocess] 最终 test_X.shape  = {test_X.shape}")
    print(f"[preprocess] 最终 train_y.shape = {train_y.shape}")
    return train_X, test_X, train_y

def get_dataloader(train_X,train_y,batch_size=32):
     dataset=TensorDataset(train_X,train_y)
     loader=DataLoader(dataset,
                       batch_size=batch_size,
                       num_workers=2,
                       pin_memory=True)
     print(f"[dataloader] batch_size={batch_size}, 共 {len(loader)} 批")
     return loader


def get_data(batch_size=32):
    train_df, test_df, all_features = load_data()
    train_X, test_X, train_y = preprocess(train_df, test_df, all_features)
    loader = get_dataloader(train_X, train_y, batch_size)
    return train_df, test_df, train_X, test_X, train_y, loader


# if __name__=="__main__":
#     get_data()