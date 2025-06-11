import csv
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

class GameDataSet(Dataset):
  def __init__(self, data):
    self.features = data[['a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp', 'h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp']]
    self.labels = data['result']

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

training_accuracies = []
testing_accuracies = []
epochs = []

for epoch in range(300):
  model.train()

  total_loss = 0
  num_batches = 0
  correct = 0
  total = 0
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

    predictions = (torch.sigmoid(outputs) > 0.5).float()
    correct += (predictions == labels).sum().item()
    total += labels.size(0)

  if True:
    print("Epoch: " + str(epoch))
    epochs.append(epoch)
    training_loss = total_loss/num_batches
    accuracy = correct / total
    training_accuracies.append(accuracy)
    print(f"Training Loss: {training_loss:.4f}, Accuracy: {accuracy:.2%}")

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
        
        accuracy = correct / total
        testing_accuracies.append(accuracy)
        print(f"Testing Loss: {total_loss / len(test_loader):.4f}, Accuracy: {correct / total:.2%}")

plot = True

if plot:
  x = epochs
  y1 = training_accuracies
  y2 = testing_accuracies

  plt.plot(x, y1, label="Training Accuracy")
  plt.plot(x, y2, label="Testing Accuracy")

  plt.xlabel("Epochs", fontsize=14)
  plt.ylabel("Accuracy", fontsize=14)
  plt.xticks(fontsize=13)
  plt.yticks(fontsize=13)
  plt.legend(fontsize=13)

  plt.show()