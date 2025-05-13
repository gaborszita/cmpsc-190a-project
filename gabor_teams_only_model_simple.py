# This program uses only the away_team and home_team information
# about a game to predict the winner

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from data_reader import game_info
from team_name_mapping import TeamNameMapping
from matplotlib import pyplot as plt

class GameInfoDataset(Dataset):
  def _one_hot_encode_str_array(self, array, team_name_mapping):
    #values = set(array)
    #name_to_int_mapping = {team_name: i for i, team_name in enumerate(values)}
    name_to_int_mapping = team_name_mapping.get_team_name_to_int_mapping()
    array_int = [name_to_int_mapping[i] for i in array]
    int_tensor = torch.tensor(array_int)
    return F.one_hot(int_tensor)

  def __init__(self, data, team_name_mapping):
    team_1 = self._one_hot_encode_str_array([elem[3] for elem in data], team_name_mapping)
    team_2 = self._one_hot_encode_str_array([elem[5] for elem in data], team_name_mapping)
    self.features = torch.cat((team_1, team_2), dim=1).float()
    print("Features size: " + str(self.features.size()))
    self.labels = torch.tensor([elem[7] for elem in data]).float()
  
  def __len__(self):
    return len(self.features)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]

split_idx = int(len(game_info) * 0.8)
game_info_train = game_info[:split_idx]
game_info_test = game_info[split_idx:]
  
team_name_mapping = TeamNameMapping([elem[3] for elem in game_info])

train_dataset = GameInfoDataset(game_info_train, team_name_mapping)
train_loader = DataLoader(train_dataset, batch_size=32)

test_dataset = GameInfoDataset(game_info_test, team_name_mapping)
test_loader = DataLoader(test_dataset, batch_size=32)

#print(game_info_train_dataset[0][0].shape[0])

#model = nn.Linear(game_info_train_dataset[0][0].shape[0], 1)
model = nn.Sequential(
  nn.Linear(train_dataset[0][0].shape[0], 1)
)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("Training model")

training_accuracies = []
testing_accuracies = []
epochs = []

for epoch in range(50):
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
  plt.legend()

  plt.show()