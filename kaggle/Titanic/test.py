import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from Mynet import ClassTitanic
from data_process import TitanicDataset,collate_fn_test

def train_model(model, train_loader, criterion, optimizer, num_epochs=25, device='cuda'):
    model = model.to(device)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
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
            torch.save(model.state_dict(), 'best_model.pth')

    print('Training complete')
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

    submission = pd.DataFrame({
        'PassengerId': ids,
        'Survived': predictions.astype(int)
    })

    submission = submission.sort_values('PassengerId')

    submission.to_csv('submission.csv', index=False)
    print('Submission file saved as submission.csv')
    return submission


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    batch_size = 32
    num_epochs = 50
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
    model.load_state_dict(torch.load('best_model.pth'))
    submission = predict_and_save(model, test_loader, device=device)
    print("\nSubmission file preview:")
    print(submission.head())


if __name__ == '__main__':
    main()
