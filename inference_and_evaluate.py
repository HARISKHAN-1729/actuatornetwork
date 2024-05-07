import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from actuator_net1 import Actuator1
from actuator_net2 import Actuator2

csv_path = 'Path to your dataset'

def inference_and_evaluate(csv_path, device, model1_path, model2_path):
    """
    This function loads and evaluates two pre-trained neural network models on a given dataset.
    
    Arguments:
    csv_path : str
        Path to the CSV file containing the dataset with columns for 'Position Error', 'Velocity', and 'Torque'.
    device : str
        The device (e.g., 'cpu' or 'cuda') on which the tensor computations will be performed.
    model1_path : str
        Path to the first model's saved state dictionary.
    model2_path : str
        Path to the second model's saved state dictionary.
    
    The function performs the following operations:
    - Reads the data from the specified CSV file.
    - Extracts input features and the true output values.
    - Converts the data into tensors and transfers them to the specified device.
    - Loads two neural network models, specifically instances of ActuatorNet2 and ActuatorNet1.
    - Sets the models to evaluation mode.
    - Performs inference using both models without gradient calculation.
    - Computes the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for the predictions from each model.
    - Writes the computed metrics to a text file.
    - Prints the computed metrics.
    - Plots a comparison of actual and predicted torque values using matplotlib.
    """
    
    # Load data from a CSV file at the specified path.
    data = pd.read_csv(csv_path)
    # Extract the input features and the true torque values.
    X = data[['Position Error', 'Velocity']].values
    y_true = data['Torque'].values
    
    # Convert the data to torch tensors and move them to the specified device.
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    # Define and load the first model, and set it to evaluation mode.
    CustomNet = ActuatorNet2().to(device)
    CustomNet.load_state_dict(torch.load(model1_path))
    CustomNet.eval()
    
    # Define and load the second model, and set it to evaluation mode.
    PapersNet = ActuatorNet1().to(device)
    PapersNet.load_state_dict(torch.load(model2_path))
    PapersNet.eval()
    
    # Perform inference with both models and compute the errors.
    with torch.no_grad():
        y_pred1_tensor = CustomNet(X_tensor)
        y_pred1 = y_pred1_tensor.cpu().numpy().flatten()
        
        y_pred2_tensor = PapersNet(X_tensor)
        y_pred2 = y_pred2_tensor.cpu().numpy().flatten()
    
    # Calculate MAE and RMSE for both models.
    mae1 = np.mean(np.abs(y_pred1 - y_true))
    rmse1 = np.sqrt(np.mean((y_pred1 - y_true)**2))
    mae2 = np.mean(np.abs(y_pred2 - y_true))
    rmse2 = np.sqrt(np.mean((y_pred2 - y_true)**2))
    
    # Write the evaluation metrics to a text file.
    with open('/content/model_metrics.txt', 'w') as f:
        f.write(f"CustomNet - Mean Absolute Error (MAE): {mae1:.4f}, Root Mean Squared Error (RMSE): {rmse1:.4f}\n")
        f.write(f"PapersNet - Mean Absolute Error (MAE): {mae2:.4f}, Root Mean Squared Error (RMSE): {rmse2:.4f}\n")
    
    # Print the evaluation metrics.
    print(f"CustomNet - Mean Absolute Error (MAE): {mae1:.4f}, Root Mean Squared Error (RMSE): {rmse1:.4f}")
    print(f"PapersNet - Mean Absolute Error (MAE): {mae2:.4f}, Root Mean Squared Error (RMSE): {rmse2:.4f}")
    
    # Plot the actual and predicted torque values.
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual Torque', color='blue', linewidth=2)
    plt.plot(y_pred1, label='Predicted Torque (CustomNet)', color='red', linestyle='--')
    plt.plot(y_pred2, label='Predicted Torque (PapersNet)', color='green', linestyle='--')
    plt.title('Comparison of Actual and Predicted Torque')
    plt.xlabel('Sample Index')
    plt.ylabel('Torque')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
inference_and_evaluate('/content/output.csv', 'cuda', '/content/custom.pth', '/content/model.pth')
