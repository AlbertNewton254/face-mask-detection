"""Configuration settings for face mask detection project."""

import os

# Class mapping for YOLO format
CLASS_MAP = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

# Reverse mapping for visualization
REVERSE_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

# Training hyperparameters
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640
PATIENCE = 7  # Early stopping patience

# Dataset splits
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Model settings
MODEL_NAME = 'yolo11n.pt'
PROJECT_DIR = 'model'
EXPERIMENT_NAME = 'face_mask_yolo'

# Training settings
OPTIMIZER = 'SGD'
LR0 = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005

# Inference settings
CONFIDENCE_THRESHOLD = 0.5

# Directory structure
DATA_DIR = 'data'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'annotations')
YOLO_DATASET_DIR = os.path.join(DATA_DIR, 'yolo_dataset')
