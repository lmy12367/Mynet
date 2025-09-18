import numpy as np
import pandas as pd

train_data = pd.read_csv('./train.csv')
print(train_data.head(10))

test_data = pd.read_csv('./test.csv')
print((test_data.head(5)))

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)
print(train_data.dtypes)
print(test_data.dtypes)

survived = train_data['Survived']
ID = train_data["PassengerId"]
train_data.drop('Survived', axis=1, inplace=True)
train_data.drop('PassengerId', axis=1, inplace=True)
train_data.drop('Ticket', axis=1, inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)
train_data.drop('Name', axis=1, inplace=True)
train_data['Age'].fillna(train_data.Age.mean())
train_data['Embarked'].fillna(train_data.Embarked.mode()[0])
print(train_data['Embarked'].isna())

test_data['Age'].fillna(test_data['Age'].mean())
test_data['Fare'].fillna(test_data['Fare'].mead())

sex_map = {'male': 0, 'female': 1}
emb_map = {'S': 1, 'C': 2, 'Q': 3}

for df in (train_data, test_data):
    df["Sex"] = df['Sex'].map(sex_map)
    df['Embarked'] = df['Embarked'].map(emb_map)
