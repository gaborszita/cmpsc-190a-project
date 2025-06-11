# uses the difference of four-factor data between teams

import csv
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np

# resources used:
# https://www.kaggle.com/code/kanncaa1/recurrent-neural-network-with-pytorch
# https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

model_type = input("Enter model type: ")

class RNNModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, layer_dim=1, output_dim=1):
        super(RNNModel, self).__init__()
        if model_type == "LSTM":
          self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        elif model_type == "GRU":
          self.rnn = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        elif model_type == "RNN":
          self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :]) 
        return output

class GameDataSet(Dataset):
  def __init__(self, data):
    self.limit = 10
    self.data = data
    self.away_stat_cols = [
      [f'a_eFGp_{i+1}', f'a_FTr_{i+1}', f'a_ORBp_{i+1}', f'a_TOVp_{i+1}']
      for i in range(self.limit)
    ]
    self.home_stat_cols = [
      [f'h_eFGp_{i+1}', f'h_FTr_{i+1}', f'h_ORBp_{i+1}', f'h_TOVp_{i+1}']
      for i in range(self.limit)
    ]
    self.labels = data['result']

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    row = self.data.iloc[idx]
    sequence = []
    for i in range(self.limit):
      away_stats = np.array([row[col] for col in self.away_stat_cols[i]])
      home_stats = np.array([row[col] for col in self.home_stat_cols[i]])
      step_features = home_stats - away_stats
      sequence.append(step_features)

    x = torch.tensor(sequence, dtype=torch.float32)
    y = torch.tensor(self.labels.iloc[idx], dtype=torch.float32)
    return x, y

if torch.backends.mps.is_available(): # mac
      device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
device = torch.device("cpu")
print("Using device: " + str(device))

team_factor_10 = pd.read_csv('cache_data_2/team_factor_individual_10.csv', index_col=0)
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

num_trains = 5
all_training_accuracies = []
all_testing_accuracies = []

for train_num in range(num_trains):
  print("Training model " + str(train_num+1))

  model = RNNModel()
  model = model.to(device)

  criterion = nn.BCEWithLogitsLoss()
  #optimizer = optim.SGD(model.parameters(), lr=0.01)
  optimizer = optim.Adam(model.parameters(), lr=0.001)

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
      inputs, labels = data[0].to(device), data[1].to(device)
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

    if epoch % 1 == 0:
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
              inputs, labels = inputs.to(device), labels.to(device)
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

  all_training_accuracies.append(training_accuracies)
  all_testing_accuracies.append(testing_accuracies)

# Convert lists to numpy arrays
all_training_accuracies = np.array(all_training_accuracies)
all_testing_accuracies = np.array(all_testing_accuracies)

# Calculate mean and standard deviation for training accuracies
mean_training_accuracies = np.mean(all_training_accuracies, axis=0)
std_training_accuracies = np.std(all_training_accuracies, axis=0)
lower_bound_training = mean_training_accuracies - std_training_accuracies
upper_bound_training = mean_training_accuracies + std_training_accuracies

# Calculate mean and standard deviation for testing accuracies
mean_testing_accuracies = np.mean(all_testing_accuracies, axis=0)
std_testing_accuracies = np.std(all_testing_accuracies, axis=0)
lower_bound_testing = mean_testing_accuracies - std_testing_accuracies
upper_bound_testing = mean_testing_accuracies + std_testing_accuracies

plot = True

if plot:
  x = epochs
  y1 = mean_training_accuracies
  y2 = mean_testing_accuracies

  # Plot mean accuracies
  plt.plot(x, y1, label="Mean Training Accuracy", color="blue")
  plt.plot(x, y2, label="Mean Testing Accuracy", color="orange")

  # Fill between lower and upper bounds for training
  plt.fill_between(
      x, 
      lower_bound_training, 
      upper_bound_training, 
      color="blue", 
      alpha=0.2, 
      label="Training Accuracy Bound 1 std dev"
  )

  # Fill between lower and upper bounds for testing
  plt.fill_between(
      x, 
      lower_bound_testing, 
      upper_bound_testing, 
      color="orange", 
      alpha=0.2, 
      label="Testing Accuracy Bound 1 std dev"
  )

  # Add labels, legend, and show the plot
  plt.xlabel("Epochs", fontsize=14)
  plt.ylabel("Accuracy", fontsize=14)
  plt.xticks(fontsize=13)
  plt.yticks(fontsize=13)
  plt.legend(fontsize=13)
  plt.title("Training and Testing Accuracy for " + model_type, fontsize=14)
  plt.grid(True)
  plt.tight_layout()
  #plt.show()
  save_path = f"../results/gabor_{model_type.lower()}_2.png"
  plt.savefig(save_path)

# Calculate peak accuracies and the epochs they occurred
peak_accuracies = []
peak_epochs = []

for run_accuracies in all_testing_accuracies:
    peak_accuracy = np.max(run_accuracies)
    peak_epoch = np.argmax(run_accuracies) + 1  # Add 1 to epoch because counting starts at 0
    peak_accuracies.append(peak_accuracy)
    peak_epochs.append(peak_epoch)

# Calculate mean and standard deviation of peak accuracies
mean_peak_accuracy = np.mean(peak_accuracies)
std_peak_accuracy = np.std(peak_accuracies)

# Calculate the average epoch of peak accuracies
average_peak_epoch = np.mean(peak_epochs)

# Prepare the output content
output_content = [
    f"Mean Peak Accuracy: {mean_peak_accuracy:.4f}",
    f"Standard Deviation of Peak Accuracy: {std_peak_accuracy:.4f}",
    f"Average Epoch of Peak Accuracy: {average_peak_epoch:.4f}",
    "",
    "Individual Peak Accuracies and Epochs:",
]

for i, (accuracy, epoch) in enumerate(zip(peak_accuracies, peak_epochs)):
    output_content.append(f"Run {i + 1}: Peak Accuracy = {accuracy:.4%}, Epoch = {epoch}")

# Save the results to a file
save_path = f"../results/gabor_{model_type.lower()}_2_results.txt"
with open(save_path, "w") as f:
    f.write("\n".join(output_content))