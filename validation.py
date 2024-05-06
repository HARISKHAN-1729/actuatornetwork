import torch
from metrics import compute_metrics

def validate_model(model, val_loader, criterion, device):
    """
    Validates the model using the validation dataset.

    Args:
    - model (torch.nn.Module): The trained model to be evaluated.
    - val_loader (DataLoader): DataLoader containing the validation data.
    - criterion (torch.nn.modules.loss): Loss function to measure the model performance.
    - device (torch.device): Device to which tensors will be sent (e.g., 'cuda' or 'cpu').

    Returns:
    - avg_loss (float): Average loss over the validation dataset.
    - avg_mae (float): Average Mean Absolute Error over the validation dataset.
    - avg_rmse (float): Average Root Mean Squared Error over the validation dataset.

    This function does not modify the model's weights, ensuring that the evaluation is unbiased.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Loss calculation
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            
            # Metric calculations
            mse, rmse, mae = compute_metrics(outputs, targets)
            total_mae += mae * inputs.size(0)
            total_rmse += rmse * inputs.size(0)
            total_samples += inputs.size(0)

    # Calculate averages of loss and metrics
    avg_loss = running_loss / total_samples
    avg_mae = total_mae / total_samples
    avg_rmse = total_rmse / total_samples

    return avg_loss, avg_mae, avg_rmse
