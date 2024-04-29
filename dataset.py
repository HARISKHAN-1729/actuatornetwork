import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch

class LogData(Dataset):
    """Custom dataset class for loading log data from text files. Each file is read into a pandas DataFrame,
    converted to numpy arrays, and stacked into a single numpy array that holds all the data.

    Attributes:
        directory (str): Directory path where log files are stored.
        data (np.array): Numpy array containing all the data from the log files.
    """
    def __init__(self, directory):
        self.data = []
        for filename in os.listdir(directory):
            if filename.endswith(".txt"): 
                file_path = os.path.join(directory, filename)
                df = pd.read_csv(file_path, delimiter=",", header=None)  
                self.data.append(df.values)
        self.data = np.vstack(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        velocity = self.data[idx, 0]
        position_error = self.data[idx, 1]
        torque = self.data[idx, 2]
        return torch.tensor([velocity, position_error], dtype=torch.float32), torch.tensor(torque, dtype=torch.float32)

def load_data():
    """Loads training and validation datasets from specified directories and creates corresponding data loaders.

    Returns:
        tuple: Contains training and validation DataLoader objects.
    """
    train_data = LogData('drive_training')
    val_data = LogData('drive_validation')
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=10, shuffle=False)
    return train_loader, val_loader
