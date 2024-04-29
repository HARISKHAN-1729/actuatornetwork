# actuatornetwork

This repository contains a PyTorch-based project designed to predict actuator torque from velocity and position error using deep learning techniques. The project is structured to facilitate easy understanding and use of the machine learning model for actuator performance, including data loading, model training, and evaluation.

## Project Structure

This project is organized into several folders and files:

- `datasets/`: Contains the `datasets.py` script which defines a custom dataset class for handling data operations.
- `models/`: Contains the `model.py` script that defines the neural network architecture.
- `training/`: Includes `training.py` for training and evaluation functions, and utilities for visualizing model predictions.
- `run.py`: The main script that ties everything together; it's used to execute the model's training and evaluation pipeline.

## Installation

To set up your environment for running this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/actuator-performance-prediction.git
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
