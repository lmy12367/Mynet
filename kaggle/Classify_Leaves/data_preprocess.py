import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
train_df=pd.read_csv("./data/train.csv")
print(train_df.head())
print(len(train_df))

print("类别数",train_df['label'].nunique())
le=LabelEncoder()
train_df['label_id']=le.fit_transform(train_df['label'])

with open('./data/label2idx.json', 'w') as f:
    json.dump({cls: idx for idx, cls in enumerate(le.classes_)},f, indent=2)
print('映射已保存到 ./data/label2idx.json')
