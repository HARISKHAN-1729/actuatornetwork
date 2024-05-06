import torch
from metrics import compute_metrics

def train_model(model, train_loader, criterion, optimizer, device):
    """
    Trains a model for one epoch over the given DataLoader.

    Args:
    - model (torch.nn.Module): The model to be trained.
    - train_loader (DataLoader): DataLoader containing the training data.
    - criterion (torch.nn.modules.loss): Loss function to measure the model performance.
    - optimizer (torch.optim.Optimizer): Optimizer to update model weights.
    - device (torch.device): Device to which tensors will be sent (e.g., 'cuda' or 'cpu').

    Returns:
    - avg_loss (float): Average loss over the training dataset.
    - avg_mae (float): Average Mean Absolute Error over the training dataset.
    - avg_rmse (float): Average Root Mean Squared Error over the training dataset.

    Each metric is calculated across all batches and then averaged.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_samples = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Loss calculation
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update running loss and metrics
        running_loss += loss.item() * inputs.size(0)
        mse, rmse, mae = compute_metrics(outputs.detach(), targets)
        total_mae += mae * inputs.size(0)
        total_rmse += rmse * inputs.size(0)
        total_samples += inputs.size(0)

    # Calculate averages of loss and metrics
    avg_loss = running_loss / total_samples
    avg_mae = total_mae / total_samples
    avg_rmse = total_rmse / total_samples

    return avg_loss, avg_mae, avg_rmse
