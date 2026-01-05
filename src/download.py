"""Module for downloading face mask detection dataset from Kaggle."""

import os
import kagglehub


def download_dataset():
    """
    Download the Face Mask Detection dataset from Kaggle.

    Returns:
        str: Path to the downloaded dataset.
    """
    print("Downloading Face Mask Detection dataset...")
    path = kagglehub.dataset_download("andrewmvd/face-mask-detection")
    print(f"Dataset downloaded to: {path}")

    # Verify download
    images_dir = os.path.join(path, "images")
    annotations_dir = os.path.join(path, "annotations")

    num_images = len(os.listdir(images_dir)) if os.path.exists(images_dir) else 0
    num_annotations = len(os.listdir(annotations_dir)) if os.path.exists(annotations_dir) else 0

    print(f"Images: {num_images}")
    print(f"Annotations: {num_annotations}")

    return path


def get_dataset_paths(base_path):
    """
    Get paths to images and annotations directories.

    Args:
        base_path: Base path to the dataset.

    Returns:
        tuple: (images_dir, annotations_dir)
    """
    images_dir = os.path.join(base_path, "images")
    annotations_dir = os.path.join(base_path, "annotations")
    return images_dir, annotations_dir


if __name__ == "__main__":
    dataset_path = download_dataset()
    images, annotations = get_dataset_paths(dataset_path)
    print(f"\nImages directory: {images}")
    print(f"Annotations directory: {annotations}")
