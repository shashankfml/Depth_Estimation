"""
Evaluation script for SPAD depth estimation models.
"""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

import config
from dataset import SPADDataset, get_transforms
from models import get_model
from utils import load_model, calculate_metrics, visualize_predictions


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on dataset.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, depths in dataloader:
            images, depths = images.to(device), depths.to(device)
            predictions = model(images)
            
            all_predictions.append(predictions)
            all_targets.append(depths)
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets)
    
    return metrics, all_predictions, all_targets


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='SPAD Depth Estimation Evaluation')
    parser.add_argument('--model', type=str, required=True,
                       choices=['unet', 'multiscale', 'depthnet'],
                       help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_split', type=str, default='val',
                       choices=['train', 'val'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions')
    parser.add_argument('--num_vis', type=int, default=4,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading {args.model} model...")
    model = get_model(args.model)
    model = load_model(model, args.checkpoint, device)
    
    # Load dataset
    transform = get_transforms()
    
    if args.data_split == 'val':
        dataset = SPADDataset(
            config.VAL_IMAGE_PATH,
            config.VAL_DEPTH_PATH,
            transform=transform
        )
    else:
        dataset = SPADDataset(
            config.TRAIN_IMAGE_PATH,
            config.TRAIN_DEPTH_PATH,
            transform=transform
        )
    
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Evaluating on {len(dataset)} {args.data_split} samples")
    
    # Evaluate model
    print("Evaluating model...")
    metrics, predictions, targets = evaluate_model(model, dataloader, device)
    
    # Print results
    print("\nEvaluation Results:")
    print("==================")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"Relative Error: {metrics['rel_error']:.6f}")
    
    # Visualize predictions if requested
    if args.visualize:
        print(f"\nVisualizing {args.num_vis} predictions...")
        visualize_predictions(model, dataloader, device, args.num_vis)


if __name__ == "__main__":
    main()
