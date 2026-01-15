# This program was original made by following along with Codemy.com tutorials on YouTube.
# Playlist here:
# https://youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1&si=qN9cA47jvIpXJ6Ez

# It has since been modified for use as a template for multiple projects

import torch # Import the PyTorch library
import torch.nn as nn # Import the neural network module from PyTorch for defining layers and loss functions
import torch.nn.functional as F # Import functional interface for operations like activation functions (ReLU)

# Create a Model Class that inherits nn.Module, the base class for all neural network modules in PyTorch
class Model(nn.Module):
  # Input layer (63 features) -->
  # Hidden Layer1 (number of neurons) --> 128
  # H2 (n) --> 64
  # H3 (n) --> 32
  # Output layer --> 6 classes
  def __init__(self, in_features=63, h1=128, h2=64, h3=32, out_features=7, dropout=0.2):
    super().__init__() # instantiate our nn.Module, initializing the parent class
    self.fc1 = nn.Linear(in_features, h1) # Define the first fully connected (linear) layer: input -> h1
    #self.bn1 = nn.BatchNorm1d(h1) # Add batch normalization after fc1
    self.fc2 = nn.Linear(h1, h2) # Define the second fully connected layer: h1 -> h2
    #self.bn2 = nn.BatchNorm1d(h2) # Add batch normalization after fc2
    self.fc3 = nn.Linear(h2, h3) # Define the third fully connected layer: h2 -> h3
    #self.bn3 = nn.BatchNorm1d(h3) # Add batch normalization after fc3
    self.out = nn.Linear(h3, out_features) # Define the output layer: h3 -> out_features (6 classes)
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
    
    x = self.out(x) # Pass through output layer (returns raw logits for CrossEntropyLoss)

    return x # Return the final output
  
# Pick a manual seed for randomization so results are reproducible
torch.manual_seed(69)
# Create an instance of our defined Model class
model = Model()



import pandas as pd # Import pandas for data manipulation (DataFrames)
import matplotlib.pyplot as plt # Import matplotlib for plotting graphs

Training_df = pd.read_csv('Mech_Lab_1/TrainingData/Train.csv') # Load the CSV data from the Train1.csv file into a pandas DataFrame

Training_df.tail()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# # Initialize the MinMaxScaler
# scaler = MinMaxScaler()

# # Scale the feature columns to [0, 1]
# Training_df[feature_columns] = scaler.fit_transform(Training_df[feature_columns])

# print("Features scaled successfully!")
# print(f"\nScaled data preview:")
# print(Training_df.head())
# print(f"\nMin values:\n{Training_df[feature_columns].min()}")
# print(f"\nMax values:\n{Training_df[feature_columns].max()}")

# Split the dataframe first to ensure alignment between arrays and Testing_df
# test_size=0.2 means 20% for testing, 80% for training
Training_df_split, Testing_df = train_test_split(Training_df, test_size=0.2, random_state=42, shuffle=True)

# Now extract X and y from the split dataframes
X_train = Training_df_split.iloc[:, :-1].values  # All columns except last
y_train = Training_df_split.iloc[:, -1].values   # Last column only

X_test = Testing_df.iloc[:, :-1].values  # All columns except last
y_test = Testing_df.iloc[:, -1].values   # Last column only

# Reset index on Testing_df for cleaner visualization later
Testing_df = Testing_df.reset_index(drop=True)

print(f"Training set: {X_train.shape} samples")
print(f"Testing set: {X_test.shape} samples")

# Convert X features to float tensors, which are the standard data type for inputs in PyTorch
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert y labels to long tensors for classification (not FloatTensor for regression)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set the criterion of model to measure the error, how far off the predictions are from the data
criterion = nn.CrossEntropyLoss() # Use Cross Entropy Loss for multi-class classification
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
  loss = criterion(y_pred, y_train) # Calculate loss between predicted and actual labels

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
  loss = criterion(y_eval, y_test) # Find the loss or error on the test set

print(f'Test Loss: {loss.item():.4f}')

# Calculate accuracy on test set
model.eval()  # Set model to evaluation mode
with torch.no_grad(): # Disable gradient calculation
  # Get predicted classes using argmax
  y_pred_classes = torch.argmax(model(X_test), dim=1)  # Returns class 0-5
  
  # Calculate accuracy
  correct = (y_pred_classes == y_test).sum().item()
  accuracy = correct / len(y_test) * 100
  
  print(f'\nTest Accuracy: {accuracy:.2f}% ({correct}/{len(y_test)} correct)')
  
  # Print first 20 predictions as examples
  print('\nSample Predictions:')
  for i in range(min(20, len(y_test))):
    print(f'{i+1}.) \t Predicted: {y_pred_classes[i].item()} \t Actual: {y_test[i].item()}')

# Save the trained model to a file for later use
torch.save(model.state_dict(), 'Mech_Lab_1/gesture_model.pth')
print("Model saved as 'gesture_model.pth'")

