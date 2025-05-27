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
import numpy as np

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

#model = nn.Linear(game_info_train_dataset[0][0].shape[0], 1)]

def train_model(dataset):
    indices = torch.randint(0, len(dataset), (len(dataset),))
    subset = torch.utils.data.Subset(dataset, indices)
    subset = train_dataset
    loader = DataLoader(subset, batch_size=32, shuffle=True)
    model = nn.Sequential(
      nn.Linear(dataset[0][0].shape[0], 1)
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epochs = 30
    
    for epoch in range(epochs):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
    return model

n_estimators = 10
models = []

for i in range(n_estimators):
    print("Training model " + str(i))
    model = train_model(train_dataset)
    models.append(model)

def evaluate_ensemble(models, test_loader):
    all_preds = []

    for model in models:
        preds = []
        model.eval()
        with torch.no_grad():
            for features, _ in test_loader:
                output = torch.sigmoid(model(features).squeeze())
                preds.append(output)
        all_preds.append(torch.cat(preds))

    ensemble_preds = torch.stack(all_preds).mean(dim=0)
    return ensemble_preds

ensemble_preds = evaluate_ensemble(models, test_loader)
true_labels = torch.cat([labels for _, labels in test_loader])

# Binary accuracy (assuming threshold of 0.5)
predicted = (ensemble_preds > 0.5).float()
accuracy = (predicted == true_labels).float().mean().item()
print(f"Bagged Ensemble Accuracy: {accuracy:.4f}")