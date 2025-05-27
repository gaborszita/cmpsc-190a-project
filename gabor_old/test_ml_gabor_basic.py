# This program uses only the away_team and home_team information
# about a game to predict the winner

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from data_reader import game_info
from team_name_mapping import TeamNameMapping

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
    self.labels = torch.tensor([elem[7] for elem in data]).float()
  
  def __len__(self):
    return len(self.features)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]

split_idx = int(len(game_info) * 0.8)
game_info_train = game_info[:split_idx]
game_info_test = game_info[split_idx:]
  
team_name_mapping = TeamNameMapping([elem[3] for elem in game_info])

game_info_train_dataset = GameInfoDataset(game_info_train, team_name_mapping)
game_info_trainloader = DataLoader(game_info_train_dataset, batch_size=32)

game_info_test_dataset = GameInfoDataset(game_info_test, team_name_mapping)
game_info_testloader = DataLoader(game_info_test_dataset, batch_size=32)

#print(game_info_train_dataset[0][0].shape[0])

#model = nn.Linear(game_info_train_dataset[0][0].shape[0], 1)
model = nn.Sequential(
  nn.Linear(game_info_train_dataset[0][0].shape[0], 128),
  nn.ReLU(),
  nn.Linear(128, 1)
)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("Training model")

for epoch in range(300):
  model.train()

  total_loss = 0
  num_batches = 0
  for data in game_info_trainloader:
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
    print("Training Loss: " + str(total_loss/num_batches))

    model.eval()
    with torch.no_grad():
      total_loss = 0
      num_batches = 0
      for data in game_info_testloader:
        inputs, labels = data[0], data[1]
        labels = labels.unsqueeze(1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        num_batches += 1
      print("Testing loss: " + str(total_loss/num_batches))

# Calculate baseline loss - we calculate how much the loss is by
# always predicting 0 and always predicting 1
print("Baseline")

def tensor_of_zeros(x):
    return torch.zeros(x.size(0), dtype=torch.float32).unsqueeze(1)

def tensor_of_ones(x):
    return torch.ones(x.size(0), dtype=torch.float32).unsqueeze(1)

total_loss = 0
num_batches = 0
for data in game_info_trainloader:
  outputs = tensor_of_zeros(inputs)
  loss = criterion(outputs, labels)
  total_loss += loss
  num_batches += 1
print("All zeros loss: " + str(total_loss/num_batches))

total_loss = 0
num_batches = 0
for data in game_info_trainloader:
  outputs = tensor_of_ones(inputs)
  loss = criterion(outputs, labels)
  total_loss += loss
  num_batches += 1
print("All ones loss: " + str(total_loss/num_batches))