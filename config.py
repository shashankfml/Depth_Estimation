"""
Configuration settings for SPAD depth estimation project.
"""

import os

# Data paths
TRAIN_IMAGE_PATH = 'ee-5179-modern-computer-vision-course-competition/competition-data/training-images'
TRAIN_DEPTH_PATH = 'ee-5179-modern-computer-vision-course-competition/competition-data/training-depths'
VAL_IMAGE_PATH = 'ee-5179-modern-computer-vision-course-competition/competition-data/validation-images'
VAL_DEPTH_PATH = 'ee-5179-modern-computer-vision-course-competition/competition-data/validation-depths'

# Training parameters
BATCH_SIZE = 8
LEARNING_RATE = 0.01
EPOCHS = 30
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.9

# Model parameters
IMAGE_SIZE = (256, 256)
INPUT_CHANNELS = 1  # Grayscale SPAD images
OUTPUT_CHANNELS = 1  # Depth maps

# Training settings
PATIENCE = 5  # Early stopping patience
DEVICE = 'cuda'  # or 'cpu'
SEED = 42

# Output settings
MODEL_SAVE_DIR = 'saved_models'
LOG_DIR = 'logs'
PLOT_DIR = 'plots'

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
