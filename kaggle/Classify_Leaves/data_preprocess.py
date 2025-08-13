import pandas as pd

train_df=pd.read_csv("./data/train.csv")
test_df=pd.read_csv("./data/test.csv")

print(train_df.head())
print(len(train_df))
print((len(test_df)))
print(train_df['label'].nunique())