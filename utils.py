"""
Utility functions for SPAD depth estimation project.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from datetime import datetime
import config


def set_seed(seed=None):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    seed = seed or config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        save_path (str, optional): Path to save plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', markersize=4)
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss', markersize=4)
    
    plt.title('Training & Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def visualize_predictions(model, dataloader, device, num_samples=4):
    """
    Visualize model predictions.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        num_samples (int): Number of samples to visualize
    """
    model.eval()
    
    with torch.no_grad():
        for i, (images, depths) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images, depths = images.to(device), depths.to(device)
            predictions = model(images)
            
            # Convert to numpy for plotting
            img = images[0].cpu().squeeze().numpy()
            true_depth = depths[0].cpu().squeeze().numpy()
            pred_depth = predictions[0].cpu().squeeze().numpy()
            
            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title('Input SPAD Image')
            axes[0].axis('off')
            
            axes[1].imshow(true_depth, cmap='viridis')
            axes[1].set_title('Ground Truth Depth')
            axes[1].axis('off')
            
            axes[2].imshow(pred_depth, cmap='viridis')
            axes[2].set_title('Predicted Depth')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()


def calculate_metrics(predictions, targets):
    """
    Calculate depth estimation metrics.
    
    Args:
        predictions (torch.Tensor): Predicted depth maps
        targets (torch.Tensor): Ground truth depth maps
        
    Returns:
        dict: Dictionary of metrics
    """
    with torch.no_grad():
        # MSE and RMSE
        mse = torch.mean((predictions - targets) ** 2)
        rmse = torch.sqrt(mse)
        
        # MAE
        mae = torch.mean(torch.abs(predictions - targets))
        
        # Relative error
        rel_error = torch.mean(torch.abs(predictions - targets) / (targets + 1e-8))
        
        return {
            'mse': mse.item(),
            'rmse': rmse.item(),
            'mae': mae.item(),
            'rel_error': rel_error.item()
        }


def save_model_info(model, save_path):
    """
    Save model architecture information.
    
    Args:
        model: PyTorch model
        save_path (str): Path to save info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = f"""
Model Information
================
Architecture: {model.__class__.__name__}
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Model Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)

Model Architecture:
{str(model)}
"""
    
    with open(save_path, 'w') as f:
        f.write(info)
    
    print(f"Model info saved to: {save_path}")


def load_model(model, checkpoint_path, device=None):
    """
    Load model from checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path (str): Path to checkpoint
        device: Device to load on
        
    Returns:
        nn.Module: Loaded model
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    
    print(f"Model loaded from: {checkpoint_path}")
    return model


def create_experiment_dir(experiment_name=None):
    """
    Create directory for experiment outputs.
    
    Args:
        experiment_name (str, optional): Name of experiment
        
    Returns:
        str: Path to experiment directory
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    exp_dir = os.path.join("experiments", experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    
    return exp_dir


def count_parameters(model):
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }
