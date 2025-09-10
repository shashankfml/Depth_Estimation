"""
Main training script for SPAD depth estimation.
"""

import argparse
import torch
from datetime import datetime
import os

import config
from dataset import get_dataloaders
from models import get_model
from train import Trainer
from utils import set_seed, plot_training_curves, save_model_info, create_experiment_dir


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='SPAD Depth Estimation Training')
    parser.add_argument('--model', type=str, default='unet', 
                       choices=['unet', 'multiscale', 'depthnet'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for experiment directory')
    parser.add_argument('--seed', type=int, default=config.SEED,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create experiment directory
    exp_dir = create_experiment_dir(args.experiment_name)
    print(f"Experiment directory: {exp_dir}")
    
    # Update config with command line arguments
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print(f"Creating {args.model} model...")
    model = get_model(args.model)
    
    # Save model info
    model_info_path = os.path.join(exp_dir, "model_info.txt")
    save_model_info(model, model_info_path)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device)
    
    # Train model
    print("Starting training...")
    history = trainer.train(args.epochs)
    
    # Plot training curves
    plot_path = os.path.join(exp_dir, "plots", "training_curves.png")
    plot_training_curves(
        history['train_losses'], 
        history['val_losses'], 
        save_path=plot_path
    )
    
    # Save final results
    results_path = os.path.join(exp_dir, "results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Experiment Results\n")
        f.write(f"==================\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Epochs trained: {len(history['train_losses'])}\n")
        f.write(f"Best validation loss: {history['best_val_loss']:.6f}\n")
        f.write(f"Final training loss: {history['train_losses'][-1]:.6f}\n")
        f.write(f"Model saved at: {history['model_path']}\n")
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    print(f"Results saved in: {exp_dir}")


if __name__ == "__main__":
    main()
