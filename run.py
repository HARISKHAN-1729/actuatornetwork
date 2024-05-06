import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from actuator_net1 import ActuatorNet1
from actuator_net2 import ActuatorNet2
from dataset_setup import load_and_prepare_data, split_data
from training import train_model
from validation import validate_model
from metrics import compute_metrics


file_path = '/content/combined_data.csv'
X_tensor, y_tensor = load_and_prepare_data(filepath)
train_loader, val_loader = split_data(X_tensor, y_tensor, train_ratio=0.9)

model1 = ActuatorNet1().to(device)
model2 = ActuatorNet2().to(device)  # Choose model as per requirement
criterion = nn.SmoothL1Loss(beta=0.3)
optimizer = optim.Adam(model1.parameters(), lr=0.0007)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 50
metrics = {
    'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [],
    'train_rmse': [], 'val_rmse': []
}

for epoch in range(num_epochs):
    # Training the model
    train_loss, train_mae, train_rmse = train_model(model, train_loader, criterion, optimizer, device)
    # Validating the model
    val_loss, val_mae, val_rmse = validate_model(model, val_loader, criterion, device)

    # Storing metrics
    metrics['train_loss'].append(train_loss)
    metrics['val_loss'].append(val_loss)
    metrics['train_mae'].append(train_mae)
    metrics['val_mae'].append(val_mae)
    metrics['train_rmse'].append(train_rmse)
    metrics['val_rmse'].append(val_rmse)

    # Clear previous output and display the plot
    clear_output(wait=True)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plotting Loss
    axs[0].plot(metrics['train_loss'], label='Training Loss')
    axs[0].plot(metrics['val_loss'], label='Validation Loss')
    axs[0].set_title('Loss over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plotting MAE
    axs[1].plot(metrics['train_mae'], label='Training MAE')
    axs[1].plot(metrics['val_mae'], label='Validation MAE')
    axs[1].set_title('MAE over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('MAE')
    axs[1].legend()

    # Plotting RMSE
    axs[2].plot(metrics['train_rmse'], label='Training RMSE')
    axs[2].plot(metrics['val_rmse'], label='Validation RMSE')
    axs[2].set_title('RMSE over Epochs')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('RMSE')
    axs[2].legend()

    plt.show()

    # Adjust learning rate
    scheduler.step()

    # Print metrics for current epoch
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}')
