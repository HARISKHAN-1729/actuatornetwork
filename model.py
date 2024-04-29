import torch
from torch import nn

class ActuatorNet(nn.Module):
    """Neural network model for predicting actuator torque from velocity and position error.

    The network consists of a series of linear layers and Softsign activations. The final layer outputs a single
    value representing the predicted torque.

    Attributes:
        layers (nn.Sequential): The sequential container of layers comprising the network.
    """
    def __init__(self):
        super(ActuatorNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.Softsign(),
            nn.Linear(32, 32),
            nn.Softsign(),
            nn.Linear(32, 32),
            nn.Softsign(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)
