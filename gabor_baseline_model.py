# Result: Data overfitting extremely to the training data

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from data_reader import game_info_detailed
import random
import matplotlib.pyplot as plt
from team_name_mapping import TeamNameMapping

class GameInfoDataset(Dataset):
  def _one_hot_encode_str_array(self, array, team_name_mapping):
    #values = set(array)
    #name_to_int_mapping = {team_name: i for i, team_name in enumerate(values)}
    name_to_int_mapping = team_name_mapping.get_team_name_to_int_mapping()
    array_int = [name_to_int_mapping[i] for i in array]
    int_tensor = torch.tensor(array_int)
    return F.one_hot(int_tensor)

  def __init__(self, data, team_name_mapping, min_vals, max_vals):
    team_1 = self._one_hot_encode_str_array([elem[1] for elem in data], team_name_mapping)
    team_2 = self._one_hot_encode_str_array([elem[2] for elem in data], team_name_mapping)
    data_rest = torch.tensor([elem[3:] for elem in data])
    #mean_vals = data_rest.mean(dim=0)
    #std_vals = data_rest.std(dim=0)
    #min_vals = data_rest.min(dim=0).values
    #max_vals = data_rest.max(dim=0).values
    #data_rest = (data_rest - mean_vals) / std_vals
    data_rest = (data_rest - min_vals) / (max_vals - min_vals)
    #self.features = torch.cat((team_1, team_2, data_rest), dim=1).float()
    self.features = data_rest
    self.labels = torch.tensor([elem[0] for elem in data]).float()
  
  def __len__(self):
    return len(self.features)
  
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]
  
split_idx = int(len(game_info_detailed) * 0.8)
print(split_idx)

team_name_mapping = TeamNameMapping([elem[1] for elem in game_info_detailed])

train_data = game_info_detailed[:split_idx]
random.shuffle(train_data)
test_data = game_info_detailed[split_idx:]

print("Training data length: " + str(len(train_data)))
print("Test data length: " + str(len(test_data)))

data_rest = torch.tensor([elem[3:] for elem in game_info_detailed])
min_vals = data_rest.min(dim=0).values
max_vals = data_rest.max(dim=0).values

train_dataset = GameInfoDataset(train_data, team_name_mapping, min_vals, max_vals)
test_dataset = GameInfoDataset(test_data, team_name_mapping, min_vals, max_vals)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

criterion = nn.BCEWithLogitsLoss()

# Calculate baseline loss - we calculate how much the loss is by
# always predicting 0 and always predicting 1
print("Baseline")

def tensor_of_zeros(x):
    return torch.zeros(x.size(0), dtype=torch.float32).unsqueeze(1)

def tensor_of_ones(x):
    return torch.ones(x.size(0), dtype=torch.float32).unsqueeze(1)

total_loss = 0
num_batches = 0
correct = 0
total = 0
for inputs, labels in test_loader:
  labels = labels.unsqueeze(1)
  outputs = tensor_of_zeros(inputs)
  loss = criterion(outputs, labels)
  total_loss += loss.item()
  num_batches += 1

  predictions = (torch.sigmoid(outputs) > 0.5).float()
  correct += (predictions == labels).sum().item()
  total += labels.size(0)
print(f"All zeros accuracy: {correct / total:.2%}")
print("All zeros loss: " + str(total_loss/num_batches))

total_loss = 0
num_batches = 0
correct = 0
total = 0
for inputs, labels in test_loader:
  labels = labels.unsqueeze(1)
  outputs = tensor_of_ones(inputs)
  loss = criterion(outputs, labels)
  total_loss += loss.item()
  num_batches += 1

  predictions = (torch.sigmoid(outputs) > 0.5).float()
  correct += (predictions == labels).sum().item()
  total += labels.size(0)
print(f"All ones accuracy: {correct / total:.2%}")
print("All ones loss: " + str(total_loss/num_batches))