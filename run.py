from training import load_data, train_model, evaluate, plot_results
from model import ActuatorNet

def main():
    """
    Main function to execute the model training and evaluation process.
    """
    # Load data
    train_loader, val_loader = load_data()
    
    # Initialize the model
    model = ActuatorNet()
    
    # Train the model
    print("Starting training...")
    train_model(model, train_loader, val_loader, epochs=20)
    print("Training completed.")
    
    # Evaluate the model
    print("Evaluating model...")
    mae, rmse = evaluate(model, val_loader)
    print(f'MAE: {mae}, RMSE: {rmse}')
    
  
if __name__ == '__main__':
    main()
