import torch
from torch import nn

class ActuatorNet2(nn.Module):
    """
    Defines a neural network for an actuator featuring ReLU activations and dropout for regularization.

    The network architecture includes:
    - Input layer: 2 input features
    - Hidden layers: Three hidden layers with 32 units each, using ReLU activation
    - Dropout: A dropout layer with a dropout rate of 0.5 after the first ReLU layer to reduce overfitting
    - Output layer: Single output unit
    """
    def __init__(self):
        super(ActuatorNet2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        Defines the computation performed at every call of this model.

        Args:
        - x (torch.Tensor): The input tensor containing the features.

        Returns:
        - torch.Tensor: The output tensor after passing through the layers.
        """
        return self.layers(x)
