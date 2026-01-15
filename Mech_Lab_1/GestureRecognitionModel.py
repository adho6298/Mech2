# This program was original made by following along with Codemy.com tutorials on YouTube.
# Playlist here:
# https://youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1&si=qN9cA47jvIpXJ6Ez

# It has since been modified for use with the NASA Turbofan Jet Engine Data Set for a Machine Learning in ME course at UC Berkeley.

import torch # Import the PyTorch library
import torch.nn as nn # Import the neural network module from PyTorch for defining layers and loss functions
import torch.nn.functional as F # Import functional interface for operations like activation functions (ReLU)

# Create a Model Class that inherits nn.Module, the base class for all neural network modules in PyTorch
class Model(nn.Module):
  # Input layer (24 features of the Engine) -->
  # Hidden Layer1 (number of neurons) --> 128
  # H2 (n) --> 64
  # H3 (n) --> 32
  # H4 (n) --> 16
  # output (1 value for RUL prediction)
  def __init__(self, in_features=24, h1=128, h2=64, h3=32, h4=16, out_features=1, dropout=0.2):
    super().__init__() # instantiate our nn.Module, initializing the parent class
    self.fc1 = nn.Linear(in_features, h1) # Define the first fully connected (linear) layer: input -> h1
    #self.bn1 = nn.BatchNorm1d(h1) # Add batch normalization after fc1
    self.fc2 = nn.Linear(h1, h2) # Define the second fully connected layer: h1 -> h2
    #self.bn2 = nn.BatchNorm1d(h2) # Add batch normalization after fc2
    self.fc3 = nn.Linear(h2, h3) # Define the third fully connected layer: h2 -> h3
    #self.bn3 = nn.BatchNorm1d(h3) # Add batch normalization after fc3
    self.fc4 = nn.Linear(h3, h4) # Define the fourth fully connected layer: h3 -> h4
    #self.bn4 = nn.BatchNorm1d(h4) # Add batch normalization after fc4
    self.out = nn.Linear(h4, out_features) # Define the output layer: h4 -> output
    #self.dropout = nn.Dropout(dropout) # Dropout layer

  # Define the forward pass, which dictates how data moves through the network
  def forward(self, x):
    x = self.fc1(x) # Pass through fc1
    #x = self.bn1(x) # Apply batch normalization
    #x = F.relu(x) # Apply ReLU activation
    x = F.gelu(x) # Apply GELU activation
    #x = self.dropout(x) # Apply dropout
    
    x = self.fc2(x) # Pass through fc2
    #x = self.bn2(x) # Apply batch normalization
    x = F.gelu(x) # Apply GELU activation
    #x = F.gelu(x) # Apply GELU activation
    #x = self.dropout(x) # Apply dropout
    
    x = self.fc3(x) # Pass through fc3
    #x = self.bn3(x) # Apply batch normalization
    x = F.gelu(x) # Apply GELU activation
    #x = F.gelu(x) # Apply GELU activation
    #x = self.dropout(x) # Apply dropout
    
    x = self.fc4(x) # Pass through fc4
    #x = self.bn4(x) # Apply batch normalization
    x = F.gelu(x) # Apply GELU activation
    #x = F.gelu(x) # Apply GELU activation
    #x = self.dropout(x) # Apply dropout
    
    x = self.out(x) # Pass result through the output layer (no activation on output for regression)

    return x # Return the final output
  
# Pick a manual seed for randomization so results are reproducible
torch.manual_seed(69)
# Create an instance of our defined Model class
model = Model()

import pandas as pd # Import pandas for data manipulation (DataFrames)
import matplotlib.pyplot as plt # Import matplotlib for plotting graphs

Training_df = pd.read_csv('Train1.csv') # Load the CSV data from the Train1.csv file into a pandas DataFrame

Training_df.tail()

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Separate the feature columns (columns 3-26, which are indices 2-25 in 0-indexed)
feature_columns = Training_df.columns[2:26]  # This gets columns from index 2 to 25 (columns 3-26)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Scale the feature columns to [0, 1]
Training_df[feature_columns] = scaler.fit_transform(Training_df[feature_columns])

print("Features scaled successfully!")
print(f"\nScaled data preview:")
print(Training_df.head())
print(f"\nMin values:\n{Training_df[feature_columns].min()}")
print(f"\nMax values:\n{Training_df[feature_columns].max()}")

# Train Test Split!  Set X, y
X = Training_df.drop(['engine', 'cycle', 'RUL'], axis=1) # Create features matrix X by dropping the target column 'variety'
y = Training_df['RUL'] # Create target vector y containing only the 'variety' column

# Convert these pandas objects to numpy arrays for compatibility with PyTorch or further processing
X = X.values
y = y.values

# Load and process the test data
Testing_df = pd.read_csv('Test1.csv')

# Scale the test data features using the SAME scaler fitted on training data
Testing_df[feature_columns] = scaler.transform(Testing_df[feature_columns])

# Separate features and target for test data
X_test_data = Testing_df.drop(['engine', 'cycle', 'RUL'], axis=1)
y_test_data = Testing_df['RUL']

# Convert to numpy arrays
X_test_data = X_test_data.values
y_test_data = y_test_data.values

# Establish Training and Testing sets
X_train = X  # Training features
y_train = y  # Training targets
X_test = X_test_data  # Testing features
y_test = y_test_data  # Testing targets

print(f"Training set: {X_train.shape} samples")
print(f"Testing set: {X_test.shape} samples")

# Convert X features to float tensors, which are the standard data type for inputs in PyTorch
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert y labels to float tensors for regression (not LongTensor for classification)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# Set the criterion of model to measure the error, how far off the predictions are from the data
criterion = nn.MSELoss() # MSELoss (Mean Squared Error) is used for regression tasks
# Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Initialize optimizer to update model weights based on gradients


# Train our model!
# Epochs? (one run thru all the training data in our network)
epochs = 1000 # Define number of training iterations
losses = [] # List to store loss values for plotting later
for i in range(epochs): # Loop through the number of epochs
  # Go forward and get a prediction
  y_pred = model.forward(X_train) # Get predicted results by passing training data through the model

  # Measure the loss/error, gonna be high at first
  loss = criterion(y_pred.squeeze(), y_train) # Calculate loss by comparing predictions (y_pred) vs actual values (y_train)

  # Keep Track of our losses
  losses.append(loss.detach().numpy()) # Detach loss from graph to save memory and convert to numpy for storing

  # print every 10 epoch
  if i % 10 == 0:
    print(f'Epoch: {i} and loss: {loss}')

  # Do some back propagation: take the error rate of forward propagation and feed it back
  # thru the network to fine tune the weights
  optimizer.zero_grad() # Clear previous gradients
  loss.backward() # Calculate gradients (backpropagation)
  optimizer.step() # Update weights using the optimizer

# Graph it out!
plt.plot(range(epochs), losses) # Plot the loss values over epochs
plt.ylabel("loss/error") # Label the y-axis
plt.xlabel('Epoch') # Label the x-axis

# Evaluate Model on Test Data Set (validate model on test set)
with torch.no_grad():  # Basically turn off back propogation (gradient calculation) to save memory and computation
  y_eval = model.forward(X_test) # X_test are features from our test set, y_eval will be predictions
  loss = criterion(y_eval.squeeze(), y_test) # Find the loss or error on the test set

print(f'Test Loss (MSE): {loss.item():.4f}')


model.eval()  # Set model to evaluation mode
correct = 0
with torch.no_grad(): # Disable gradient calculation
  for i, data in enumerate(X_test): # Iterate over test data
    y_val = model.forward(data.unsqueeze(0)) # Add batch dimension: (24,) -> (1, 24)

    # Will tell us what RUL our network predicts
    print(f'{i+1}.) \t Predicted: {y_val.item():.2f} \t Actual: {y_test[i].item():.2f}')

# Calculate accuracy metrics for regression
with torch.no_grad():
  y_pred_all = model.forward(X_test)
  mae = torch.mean(torch.abs(y_pred_all.squeeze() - y_test))
  print(f'\nMean Absolute Error: {mae.item():.2f}')

  # Generate predictions for all test data
model.eval()
with torch.no_grad():
    y_predictions = model.forward(X_test).squeeze().numpy()

# Get unique engines and randomly select 8
unique_engines = Testing_df['engine'].unique()
np.random.seed(42)  # For reproducibility
selected_engines = np.random.choice(unique_engines, size=min(8, len(unique_engines)), replace=False)

# Create a figure with 8 subplots (2 rows x 4 columns)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Neural Network - Actual vs Predicted RUL for Random Engines', fontsize=16, y=1.00)

# Flatten axes array for easier iteration
axes = axes.flatten()

# Plot each selected engine
for idx, engine_num in enumerate(selected_engines):
    # Filter for current engine
    engine_mask = Testing_df['engine'] == engine_num
    engine_df = Testing_df[engine_mask]
    
    # Extract cycle data and RUL values
    X_og = engine_df['cycle'].values
    Y_og = engine_df['RUL'].values
    
    # Get predictions for current engine
    y_pred_engine = y_predictions[engine_mask]
    
    # Plot on the corresponding subplot
    axes[idx].scatter(X_og, Y_og, label='Actual RUL', color='blue', alpha=0.7, marker='x')
    axes[idx].scatter(X_og, y_pred_engine, label='Predicted RUL', color='red', alpha=0.7)
    axes[idx].set_title(f'Engine {engine_num}')
    axes[idx].set_xlabel('Cycle Count')
    axes[idx].set_ylabel('RUL')
    axes[idx].set_ylim(0, 260)  # Set y-axis limits from 0 to 260
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)
    
    # Calculate and display MAE for this engine
    mae_engine = np.mean(np.abs(Y_og - y_pred_engine))
    axes[idx].text(0.05, 0.95, f'MAE: {mae_engine:.2f}', 
                   transform=axes[idx].transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=8)

plt.tight_layout()
plt.show()

# Print summary statistics for all selected engines
print("Summary Statistics for Selected Engines:")
print("="*60)
for engine_num in selected_engines:
    engine_mask = Testing_df['engine'] == engine_num
    Y_og = Testing_df[engine_mask]['RUL'].values
    y_pred_engine = y_predictions[engine_mask]
    
    mae = np.mean(np.abs(Y_og - y_pred_engine))
    errors = (Y_og - y_pred_engine) / Y_og * 100
    avg_error = np.mean(np.abs(errors))
    
    print(f"Engine {engine_num}: MAE = {mae:.2f} cycles, Avg Error = {avg_error:.2f}%")


# Generate predictions for engines with the 8 lowest final RUL values
model.eval()
with torch.no_grad():
    y_predictions = model.forward(X_test).squeeze().numpy()

# Get final RUL for each engine and sort to find the 8 lowest
engine_final_rul = []
for engine_num in Testing_df['engine'].unique():
    engine_data = Testing_df[Testing_df['engine'] == engine_num]
    last_rul = engine_data['RUL'].iloc[-1]  # Get the last RUL value
    engine_final_rul.append((engine_num, last_rul))

# Sort by final RUL (ascending) and select the 8 engines with lowest final RUL
engine_final_rul.sort(key=lambda x: x[1])
selected_engines = [engine_num for engine_num, _ in engine_final_rul[:8]]

print(f"8 Engines with lowest final RUL:")
for engine_num, final_rul in engine_final_rul[:8]:
    print(f"  Engine {engine_num}: Final RUL = {final_rul:.1f}")

# Create a figure with 8 subplots (2 rows x 4 columns)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Neural Network - 8 Engines with Lowest Final RUL', fontsize=16, y=1.00)

# Flatten axes array for easier iteration
axes = axes.flatten()

# Plot each selected engine
for idx, engine_num in enumerate(selected_engines):
    # Filter for current engine
    engine_mask = Testing_df['engine'] == engine_num
    engine_df = Testing_df[engine_mask]
    
    # Extract cycle data and RUL values
    X_og = engine_df['cycle'].values
    Y_og = engine_df['RUL'].values
    
    # Get predictions for current engine
    y_pred_engine = y_predictions[engine_mask]
    
    # Get final RUL value
    final_rul = Y_og[-1]
    
    # Plot on the corresponding subplot
    axes[idx].scatter(X_og, Y_og, label='Actual RUL', color='blue', alpha=0.7, marker='x')
    axes[idx].scatter(X_og, y_pred_engine, label='Predicted RUL', color='red', alpha=0.7)
    axes[idx].set_title(f'Engine {engine_num} (Final RUL: {final_rul:.1f})')
    axes[idx].set_xlabel('Cycle Count')
    axes[idx].set_ylabel('RUL')
    axes[idx].set_ylim(0, 260)  # Set y-axis limits from 0 to 260
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)
    
    # Calculate and display MAE for this engine
    mae_engine = np.mean(np.abs(Y_og - y_pred_engine))
    axes[idx].text(0.05, 0.95, f'MAE: {mae_engine:.2f}', 
                   transform=axes[idx].transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=8)

plt.tight_layout()
plt.show()

# Print summary statistics for all selected engines
print("\nSummary Statistics for 8 Engines with Lowest Final RUL:")
print("="*60)
for engine_num in selected_engines:
    engine_mask = Testing_df['engine'] == engine_num
    engine_df = Testing_df[engine_mask]
    Y_og = engine_df['RUL'].values
    y_pred_engine = y_predictions[engine_mask]
    final_rul = Y_og[-1]
    
    mae = np.mean(np.abs(Y_og - y_pred_engine))
    errors = (Y_og - y_pred_engine) / Y_og * 100
    avg_error = np.mean(np.abs(errors))
    
    print(f"Engine {engine_num} (Final RUL: {final_rul:.1f}): MAE = {mae:.2f} cycles, Avg Error = {avg_error:.2f}%")
