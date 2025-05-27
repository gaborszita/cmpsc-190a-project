import csv
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class GameDataSet(Dataset):
  def __init__(self, data):
    self.features = data[['a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp', 'h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp']]
    self.labels = data['result']
    print(self.features.shape)

  def __len__(self):
    return len(self.features)
  
  def __getitem__(self, idx):
    x = torch.tensor(self.features.iloc[idx].values, dtype=torch.float32)
    y = torch.tensor(self.labels.iloc[idx], dtype=torch.float32)
    return x, y

team_factor_10 = pd.read_csv('NBA-Prediction-Modeling/data/team_factor_10.csv', index_col=0)
team_factor_10.dropna(inplace=True)

print("Data length: " + str(len(team_factor_10)))

split_idx = int(len(team_factor_10) * 0.8)
train_data = team_factor_10[:split_idx]
test_data = team_factor_10[split_idx:]

train_dataset = GameDataSet(train_data)
test_dataset = GameDataSet(test_data)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


model = nn.Sequential(
  nn.Linear(train_dataset[0][0].shape[0], 1),
)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("Training model")

training_losses = []
testing_losses = []

for epoch in range(200):
  model.train()

  total_loss = 0
  num_batches = 0
  for data in train_loader:
    inputs, labels = data[0], data[1]
    labels = labels.unsqueeze(1)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    num_batches += 1

  if epoch % 20 == 0:
    training_loss = total_loss/num_batches
    training_losses.append(training_loss)
    print("Training Loss: " + str(training_loss))

    model.eval()
    with torch.no_grad():
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            labels = labels.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        print(f"Testing Loss: {total_loss / len(test_loader):.4f}, Accuracy: {correct / total:.2%}")