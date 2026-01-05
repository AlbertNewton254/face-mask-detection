"""Main orchestration script for face mask detection pipeline."""

import os
import argparse
import random

from download import download_dataset, get_dataset_paths
from converter import convert_voc_to_yolo
from train import train_model, get_best_model_path
from evaluate import evaluate_model
from visualize import plot_class_distribution, predict_and_visualize
from config import (
    DATA_DIR, IMAGES_DIR, ANNOTATIONS_DIR, YOLO_DATASET_DIR,
    CONFIDENCE_THRESHOLD
)


def setup_data(force_download=False):
    """
    Download and prepare the dataset.

    Args:
        force_download: If True, force re-download even if data exists.

    Returns:
        tuple: (images_dir, annotations_dir)
    """
    print("\n" + "="*60)
    print("STEP 1: DATA SETUP")
    print("="*60)

    # Check if data already exists locally
    if os.path.exists(IMAGES_DIR) and os.path.exists(ANNOTATIONS_DIR) and not force_download:
        print("Data already exists locally. Skipping download.")
        return IMAGES_DIR, ANNOTATIONS_DIR

    # Download dataset
    dataset_path = download_dataset()
    images_dir, annotations_dir = get_dataset_paths(dataset_path)

    return images_dir, annotations_dir


def convert_dataset(images_dir, annotations_dir):
    """
    Convert dataset to YOLO format.

    Args:
        images_dir: Path to images directory.
        annotations_dir: Path to annotations directory.

    Returns:
        tuple: (dataset_dir, yaml_path)
    """
    print("\n" + "="*60)
    print("STEP 2: CONVERT TO YOLO FORMAT")
    print("="*60)

    # Check if already converted
    yaml_path = os.path.join(YOLO_DATASET_DIR, 'data.yaml')
    if os.path.exists(yaml_path):
        print(f"YOLO dataset already exists at: {YOLO_DATASET_DIR}")
        return YOLO_DATASET_DIR, yaml_path

    # Convert to YOLO format
    dataset_dir, yaml_path = convert_voc_to_yolo(
        images_dir,
        annotations_dir,
        YOLO_DATASET_DIR
    )

    return dataset_dir, yaml_path


def visualize_dataset(dataset_dir):
    """
    Visualize dataset statistics.

    Args:
        dataset_dir: Path to YOLO dataset directory.
    """
    print("\n" + "="*60)
    print("STEP 3: VISUALIZE DATASET")
    print("="*60)

    plot_class_distribution(dataset_dir)


def train_pipeline(yaml_path):
    """
    Train the model.

    Args:
        yaml_path: Path to YOLO data configuration file.

    Returns:
        str: Path to best model weights.
    """
    print("\n" + "="*60)
    print("STEP 4: TRAIN MODEL")
    print("="*60)

    # Check if model already exists
    best_model_path = get_best_model_path()
    if os.path.exists(best_model_path):
        print(f"Trained model already exists at: {best_model_path}")
        response = input("Do you want to retrain? (y/n): ")
        if response.lower() != 'y':
            return best_model_path

    # Train model
    model = train_model(yaml_path)

    return best_model_path


def evaluate_pipeline(model_path, yaml_path):
    """
    Evaluate the trained model.

    Args:
        model_path: Path to model weights.
        yaml_path: Path to YOLO data configuration file.

    Returns:
        dict: Evaluation metrics.
    """
    print("\n" + "="*60)
    print("STEP 5: EVALUATE MODEL")
    print("="*60)

    results = evaluate_model(model_path, yaml_path, split='test')

    return results


def visualize_predictions(model_path, dataset_dir, annotations_dir, num_samples=5):
    """
    Visualize predictions on test images.

    Args:
        model_path: Path to model weights.
        dataset_dir: Path to YOLO dataset directory.
        annotations_dir: Path to original annotations directory.
        num_samples: Number of samples to visualize.
    """
    print("\n" + "="*60)
    print("STEP 6: VISUALIZE PREDICTIONS")
    print("="*60)

    from ultralytics import YOLO

    # Load model
    model = YOLO(model_path)

    # Get test images
    test_images_dir = os.path.join(dataset_dir, "images", "test")
    test_images = [
        os.path.join(test_images_dir, f)
        for f in os.listdir(test_images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Sample random images
    random.seed(42)
    sample_images = random.sample(test_images, min(num_samples, len(test_images)))

    print(f"\nVisualizing predictions on {len(sample_images)} test images...\n")

    for img_path in sample_images:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        annotation_path = os.path.join(annotations_dir, base_name + ".xml")

        predict_and_visualize(
            img_path,
            model,
            annotation_path if os.path.exists(annotation_path) else None,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )


def run_full_pipeline(force_download=False, skip_train=False, skip_viz=False):
    """
    Run the complete face mask detection pipeline.

    Args:
        force_download: Force re-download of dataset.
        skip_train: Skip training if model exists.
        skip_viz: Skip visualization steps.
    """
    print("\n" + "="*60)
    print("FACE MASK DETECTION PIPELINE")
    print("="*60)

    # Step 1: Setup data
    images_dir, annotations_dir = setup_data(force_download)

    # Step 2: Convert to YOLO format
    dataset_dir, yaml_path = convert_dataset(images_dir, annotations_dir)

    # Step 3: Visualize dataset
    if not skip_viz:
        visualize_dataset(dataset_dir)

    # Step 4: Train model
    if not skip_train:
        model_path = train_pipeline(yaml_path)
    else:
        model_path = get_best_model_path()
        if not os.path.exists(model_path):
            print("ERROR: No trained model found. Please run training first.")
            return

    # Step 5: Evaluate model
    results = evaluate_pipeline(model_path, yaml_path)

    # Step 6: Visualize predictions
    if not skip_viz:
        visualize_predictions(model_path, dataset_dir, annotations_dir)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nModel path: {model_path}")
    print(f"Dataset path: {dataset_dir}")
    print(f"\nFinal metrics:")
    print(f"  mAP50: {results['mAP50']:.4f}")
    print(f"  mAP50-95: {results['mAP50-95']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")


def main():
    """Main entry point with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Face Mask Detection Pipeline using YOLOv11"
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'download', 'convert', 'train', 'evaluate', 'visualize'],
        default='full',
        help='Pipeline mode to run'
    )

    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download of dataset'
    )

    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip training if model exists'
    )

    parser.add_argument(
        '--skip-viz',
        action='store_true',
        help='Skip visualization steps'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to model weights (for evaluate/visualize modes)'
    )

    parser.add_argument(
        '--yaml-path',
        type=str,
        help='Path to YOLO data config (for train/evaluate modes)'
    )

    args = parser.parse_args()

    if args.mode == 'full':
        run_full_pipeline(args.force_download, args.skip_train, args.skip_viz)

    elif args.mode == 'download':
        images_dir, annotations_dir = setup_data(args.force_download)
        print(f"\nImages: {images_dir}")
        print(f"Annotations: {annotations_dir}")

    elif args.mode == 'convert':
        images_dir, annotations_dir = setup_data(args.force_download)
        dataset_dir, yaml_path = convert_dataset(images_dir, annotations_dir)
        print(f"\nDataset: {dataset_dir}")
        print(f"Config: {yaml_path}")

    elif args.mode == 'train':
        if not args.yaml_path:
            yaml_path = os.path.join(YOLO_DATASET_DIR, 'data.yaml')
        else:
            yaml_path = args.yaml_path

        if not os.path.exists(yaml_path):
            print(f"ERROR: Data config not found: {yaml_path}")
            print("Please run conversion first or provide --yaml-path")
            return

        model_path = train_pipeline(yaml_path)
        print(f"\nModel saved to: {model_path}")

    elif args.mode == 'evaluate':
        if not args.model_path:
            model_path = get_best_model_path()
        else:
            model_path = args.model_path

        if not args.yaml_path:
            yaml_path = os.path.join(YOLO_DATASET_DIR, 'data.yaml')
        else:
            yaml_path = args.yaml_path

        if not os.path.exists(model_path):
            print(f"ERROR: Model not found: {model_path}")
            return

        if not os.path.exists(yaml_path):
            print(f"ERROR: Data config not found: {yaml_path}")
            return

        evaluate_pipeline(model_path, yaml_path)

    elif args.mode == 'visualize':
        if not args.model_path:
            model_path = get_best_model_path()
        else:
            model_path = args.model_path

        if not os.path.exists(model_path):
            print(f"ERROR: Model not found: {model_path}")
            return

        # Get data paths
        images_dir, annotations_dir = setup_data()
        dataset_dir = YOLO_DATASET_DIR

        visualize_predictions(model_path, dataset_dir, annotations_dir)


if __name__ == "__main__":
    main()
