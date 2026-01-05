"""Module for evaluating YOLO face mask detection model."""

import os
import torch
from ultralytics import YOLO

from config import BATCH_SIZE, IMG_SIZE, REVERSE_CLASS_MAP


def load_model_for_evaluation(model_path):
    """
    Load a trained YOLO model for evaluation.

    Args:
        model_path: Path to the model weights file.

    Returns:
        YOLO: Loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    return model


def evaluate_model(model_path, data_yaml_path, split='test', batch_size=None, img_size=None):
    """
    Evaluate a trained YOLO model on a dataset split.

    Args:
        model_path: Path to the trained model weights.
        data_yaml_path: Path to the YOLO data configuration file.
        split: Dataset split to evaluate on ('train', 'val', or 'test').
        batch_size: Batch size for evaluation (default: BATCH_SIZE from config).
        img_size: Image size for evaluation (default: IMG_SIZE from config).

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    if img_size is None:
        img_size = IMG_SIZE

    print("=" * 60)
    print(f"EVALUATION ON {split.upper()} SET")
    print("=" * 60)

    # Load model
    model = load_model_for_evaluation(model_path)

    # Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Validate on specified split
    metrics = model.val(
        data=data_yaml_path,
        split=split,
        batch=batch_size,
        imgsz=img_size,
        device=device
    )

    # Extract and display metrics
    print(f"\n{split.capitalize()} Set Metrics:")
    print(f"{'='*60}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"{'='*60}")

    # Per-class metrics
    if hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > 0:
        print(f"\nPer-Class AP50:")
        class_names = list(REVERSE_CLASS_MAP.values())
        for i, (name, ap) in enumerate(zip(class_names, metrics.box.ap50)):
            print(f"  {name}: {ap:.4f}")
        print(f"{'='*60}")

    # Return metrics as dictionary
    results = {
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'per_class_ap50': {}
    }

    if hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > 0:
        class_names = list(REVERSE_CLASS_MAP.values())
        for i, (name, ap) in enumerate(zip(class_names, metrics.box.ap50)):
            results['per_class_ap50'][name] = float(ap)

    return results


def run_inference(model_path, image_path, confidence_threshold=0.5):
    """
    Run inference on a single image.

    Args:
        model_path: Path to the trained model weights.
        image_path: Path to the input image.
        confidence_threshold: Confidence threshold for detections.

    Returns:
        Results object from YOLO inference.
    """
    model = load_model_for_evaluation(model_path)

    results = model.predict(
        source=image_path,
        conf=confidence_threshold,
        verbose=False
    )

    return results[0]


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        model_path = sys.argv[1]
        data_yaml = sys.argv[2]
        split = sys.argv[3] if len(sys.argv) > 3 else 'test'

        print(f"Evaluating model: {model_path}")
        print(f"Data config: {data_yaml}")
        print(f"Split: {split}\n")

        results = evaluate_model(model_path, data_yaml, split)
    else:
        print("Usage: python evaluate.py <model_path> <data_yaml_path> [split]")
