import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def load_and_prepare_data(filepath):
    """
    Loads data from a CSV file and prepares tensors for the model training.

    Args:
    - filepath (str): Path to the CSV file.

    Returns:
    - X_tensor (torch.Tensor): Tensor of input features.
    - y_tensor (torch.Tensor): Tensor of target variable.
    """
    data = pd.read_csv(filepath)
    X = data[['Position Error', 'Velocity']].values
    y = data['Torque'].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    return X_tensor, y_tensor

def split_data(X_tensor, y_tensor, train_ratio=0.9):
    """
    Splits the data into training and validation datasets.

    Args:
    - X_tensor (torch.Tensor): Tensor of input features.
    - y_tensor (torch.Tensor): Tensor of target variable.
    - train_ratio (float): Fraction of data to be used for training.

    Returns:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    """
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader
