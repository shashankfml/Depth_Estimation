"""
Training utilities for SPAD depth estimation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import os
import config


class Trainer:
    """Training class for depth estimation models."""
    
    def __init__(self, model, train_loader, val_loader, device=None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Logging
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_file = os.path.join(config.LOG_DIR, f"training_log_{timestamp}.txt")
        self.model_path = os.path.join(config.MODEL_SAVE_DIR, f"best_model_{timestamp}.pth")
        
    def write_log(self, message):
        """Write message to log file and print."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (images, depths) in enumerate(self.train_loader):
            images, depths = images.to(self.device), depths.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, depths)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        return running_loss / len(self.train_loader)
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for images, depths in self.val_loader:
                images, depths = images.to(self.device), depths.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, depths)
                running_loss += loss.item()
                
        return running_loss / len(self.val_loader)
    
    def save_best_model(self, val_loss):
        """Save model if validation loss improved."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(self.model.state_dict(), self.model_path)
            self.write_log(f"New best model saved with val loss: {val_loss:.4f}")
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False
    
    def train(self, epochs=None):
        """
        Train the model.
        
        Args:
            epochs (int): Number of epochs to train
            
        Returns:
            dict: Training history
        """
        epochs = epochs or config.EPOCHS
        
        self.write_log(f"Starting training for {epochs} epochs")
        self.write_log(f"Device: {self.device}")
        self.write_log(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            # Record losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Log progress
            self.write_log(f"Epoch {epoch+1}/{epochs}, "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            improved = self.save_best_model(val_loss)
            
            # Early stopping
            if self.patience_counter >= config.PATIENCE:
                self.write_log(f"Early stopping after {epoch+1} epochs")
                break
                
            if not improved:
                self.write_log(f"No improvement for {self.patience_counter} epoch(s)")
        
        self.write_log(f"Training completed. Best val loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'model_path': self.model_path
        }


def get_optimizer(model, optimizer_name='sgd'):
    """
    Get optimizer for model.
    
    Args:
        model: PyTorch model
        optimizer_name (str): Name of optimizer ('sgd', 'adam')
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    if optimizer_name.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE * 0.1,  # Lower LR for Adam
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_loss_function(loss_name='mse'):
    """
    Get loss function by name.
    
    Args:
        loss_name (str): Name of loss function ('mse', 'l1', 'smooth_l1')
        
    Returns:
        nn.Module: Loss function
    """
    losses = {
        'mse': nn.MSELoss(),
        'l1': nn.L1Loss(),
        'smooth_l1': nn.SmoothL1Loss()
    }
    
    if loss_name not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(losses.keys())}")
    
    return losses[loss_name]
