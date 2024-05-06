import torch
from torch import nn, optim

from actuator_net1 import ActuatorNet1
from actuator_net2 import ActuatorNet2

def setup_models_and_optimizers():
    """
    Sets up the models, optimizers, and learning rate schedulers for training.

    Returns:
    - model1 (torch.nn.Module): Instance of ActuatorNet1.
    - model2 (torch.nn.Module): Instance of ActuatorNet2.
    - optimizer1 (torch.optim.Optimizer): Adam optimizer for model1.
    - optimizer2 (torch.optim.Optimizer): Adam optimizer for model2.
    - scheduler1 (torch.optim.lr_scheduler): Learning rate scheduler for optimizer1.
    - scheduler2 (torch.optim.lr_scheduler): Learning rate scheduler for optimizer2.
    """
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize models and move them to the configured device
    model1 = ActuatorNet1().to(device)
    model2 = ActuatorNet2().to(device)

    # Loss function configuration
    criterion = nn.SmoothL1Loss(beta=0.3)

    # Optimizers configuration
    optimizer1 = optim.Adam(model1.parameters(), lr=0.0007)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.0007)

    # Learning rate schedulers configuration
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=10, gamma=0.1)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=10, gamma=0.1)

    return model1, model2, optimizer1, optimizer2, scheduler1, scheduler2

