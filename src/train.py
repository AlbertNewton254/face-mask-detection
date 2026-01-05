"""Module for training YOLO face mask detection model."""

import os
import torch
from ultralytics import YOLO

from config import (
    MODEL_NAME, EPOCHS, BATCH_SIZE, IMG_SIZE, PATIENCE,
    PROJECT_DIR, EXPERIMENT_NAME, OPTIMIZER, LR0, MOMENTUM, WEIGHT_DECAY
)


def get_device():
    """
    Get the best available device (CUDA if available, otherwise CPU).

    Returns:
        str: Device name ('cuda' or 'cpu').
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def load_model(model_path=None):
    """
    Load a YOLO model.

    Args:
        model_path: Path to model weights. If None, loads default pretrained model.

    Returns:
        YOLO: Loaded YOLO model.
    """
    if model_path is None:
        model_path = MODEL_NAME

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    return model


def train_model(data_yaml_path, model_path=None, epochs=None, batch_size=None,
                img_size=None, patience=None, device=None):
    """
    Train a YOLO model on the face mask detection dataset.

    Args:
        data_yaml_path: Path to the YOLO data configuration file.
        model_path: Path to pretrained model (default: MODEL_NAME from config).
        epochs: Number of training epochs (default: EPOCHS from config).
        batch_size: Training batch size (default: BATCH_SIZE from config).
        img_size: Input image size (default: IMG_SIZE from config).
        patience: Early stopping patience (default: PATIENCE from config).
        device: Device to train on (default: auto-detect).

    Returns:
        YOLO: Trained model.
    """
    # Use defaults from config if not specified
    if epochs is None:
        epochs = EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE
    if img_size is None:
        img_size = IMG_SIZE
    if patience is None:
        patience = PATIENCE
    if device is None:
        device = get_device()

    print("\nTraining Configuration:")
    print(f"{'='*60}")
    print(f"Data config: {data_yaml_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Early stopping patience: {patience}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Load model
    model = load_model(model_path)

    # Train
    print("Starting training...")
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=patience,
        save=True,
        device=device,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer=OPTIMIZER,
        lr0=LR0,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        verbose=True,
        plots=True
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best model saved to: {os.path.join(PROJECT_DIR, EXPERIMENT_NAME, 'weights', 'best.pt')}")
    print(f"Last model saved to: {os.path.join(PROJECT_DIR, EXPERIMENT_NAME, 'weights', 'last.pt')}")

    return model


def get_best_model_path():
    """
    Get the path to the best trained model.

    Returns:
        str: Path to best.pt weights file.
    """
    return os.path.join(PROJECT_DIR, EXPERIMENT_NAME, 'weights', 'best.pt')


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        yaml_path = sys.argv[1]
        print(f"Training model with config: {yaml_path}")
        train_model(yaml_path)
    else:
        print("Usage: python train.py <data_yaml_path>")
