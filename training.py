import torch
from torch.utils.data import DataLoader
from torch import optim, nn
import numpy as np
import matplotlib.pyplot as plt

from dataset import LogData
from model import ActuatorNet

def load_data():
    """
    Loads training and validation data using the LogData class, and prepares DataLoader instances for both.

    Returns:
        tuple: A tuple containing the DataLoader instances for training and validation datasets.
    """
    train_data = LogData('drive_training')
    val_data = LogData('drive_validation')
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=10, shuffle=False)
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, epochs=10):
    """
    Trains the neural network model on the training data and evaluates it on the validation data.

    Args:
        model (ActuatorNet): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        epochs (int): Number of training epochs.

    Prints the validation loss for each epoch.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                total_val_loss += val_loss.item()
            print(f'Epoch {epoch+1}, Val Loss: {total_val_loss / len(val_loader)}')

def evaluate(model, loader):
    """
    Evaluates the model using the provided DataLoader, calculating the Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

    Args:
        model (ActuatorNet): The model to evaluate.
        loader (DataLoader): DataLoader containing the dataset to evaluate against.

    Returns:
        tuple: A tuple containing the MAE and RMSE of the model's predictions.
    """
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            predictions.extend(outputs.numpy())
            actuals.extend(targets.numpy())
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    return mae, rmse

def plot_results(predictions, actuals):
    """
    Plots the actual vs predicted torque values to visualize the model's performance.

    Args:
        predictions (array): Predicted torque values.
        actuals (array): Actual torque values.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(actuals, label='Actual Torque')
    plt.plot(predictions, label='Predicted Torque')
    plt.title('Comparison of Actual and Predicted Torque')
    plt.xlabel('Sample Index')
    plt.ylabel('Torque')
    plt.legend()
    plt.show()
