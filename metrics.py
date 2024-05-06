import torch
from torch import nn

def compute_metrics(outputs, targets):
    """
    Computes the MSE, RMSE, and MAE between the outputs of a model and the actual targets.

    Args:
    - outputs (torch.Tensor): The tensor containing the predicted values from the model.
    - targets (torch.Tensor): The tensor containing the actual values.

    Returns:
    - mse (float): Mean Squared Error.
    - rmse (float): Root Mean Squared Error.
    - mae (float): Mean Absolute Error.

    Each of these metrics is returned as a single float value, providing a measure of the accuracy of the predictions.
    """
    # Calculate Mean Squared Error
    mse = nn.MSELoss()(outputs, targets)
    
    # Calculate Root Mean Squared Error
    rmse = torch.sqrt(mse)
    
    # Calculate Mean Absolute Error
    mae = nn.L1Loss()(outputs, targets)
    
    return mse.item(), rmse.item(), mae.item()
