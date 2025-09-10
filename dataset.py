"""
Dataset module for SPAD depth estimation.
Handles loading and preprocessing of SPAD images and depth maps.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config


class SPADDataset(Dataset):
    """Dataset class for SPAD images and corresponding depth maps."""
    
    def __init__(self, image_dir, depth_dir, transform=None):
        """
        Initialize SPAD dataset.
        
        Args:
            image_dir (str): Directory containing SPAD images
            depth_dir (str): Directory containing depth maps
            transform (callable, optional): Transform to apply to images
        """
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, depth) tensors
        """
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        depth_path = os.path.join(self.depth_dir, self.image_filenames[idx])

        # Load grayscale SPAD image and depth map
        image = Image.open(image_path).convert("L")
        depth = Image.open(depth_path).convert("L")

        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)

        return image, depth


def get_transforms():
    """Get standard transforms for SPAD data."""
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor()
    ])


def get_dataloaders():
    """
    Create train and validation dataloaders.
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    transform = get_transforms()
    
    train_dataset = SPADDataset(
        image_dir=config.TRAIN_IMAGE_PATH,
        depth_dir=config.TRAIN_DEPTH_PATH,
        transform=transform
    )
    
    val_dataset = SPADDataset(
        image_dir=config.VAL_IMAGE_PATH,
        depth_dir=config.VAL_DEPTH_PATH,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False
    )
    
    return train_loader, val_loader
