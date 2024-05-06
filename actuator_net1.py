import torch
from torch import nn

class ActuatorNet1(nn.Module):
    """
    Defines a neural network for an actuator with multiple layers using the Softsign activation function.

    The network architecture includes:
    - Input layer: 2 input features
    - Hidden layers: Four hidden layers with 32 units each, using Softsign activation
    - Output layer: Single output unit
    """
    def __init__(self):
        super(ActuatorNet1, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.Softsign(),
            nn.Linear(32, 32),
            nn.Softsign(),
            nn.Linear(32, 32),
            nn.Softsign(),
            nn.Linear(32, 32),
            nn.Softsign(),
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
