import json

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

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
