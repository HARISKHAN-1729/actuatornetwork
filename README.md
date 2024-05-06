# Actuator Neural Network Training

This project aims to train neural network models for actuator control using PyTorch. It includes scripts for data preparation, model setup, training, validation, and evaluation.



## Project Structure

This project is organized into several folders and files:

- `datapreparation/`: Contains the  script which defines a custom dataset functions for handling data operations.
- `model/`: Contains the `model. pth` weights for each network.
- `training.py` for training and evaluation functions, and utilities for visualizing model predictions.
- `actuator_net1.py`: Contains an NN class described in paper "Learning agile and dynamic motor skills for legged robots"
- `actuator_net2.py`: Custom model.
- `model_setup.py`; Sets up models, optimizers, and learning rate schedulers for training.
- `Validation.py`: Contains functions for validating the models.
- `metrics.py`; Contains functions for computing evaluation metrics.
- `run.py`: The main script that ties everything together; it's used to execute the model's training and evaluation pipeline.

## Installation

To set up your environment for running this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/HARISKHAN-1729/actuatornetwork.git
   cd actuatornetwork

2. Create and activate a virtual environment (optional but recommended):
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required dependencies:
   ```bash
    pip install -r requirements.txt

4. Usage:
   ```bash
    python run.py
