"""Module for converting Pascal VOC annotations to YOLO format."""

import os
import shutil
import yaml
from sklearn.model_selection import train_test_split

from parser import parse_voc_annotation, get_image_dimensions, convert_bbox_to_yolo
from config import CLASS_MAP, RANDOM_SEED


def create_yolo_directory_structure(output_dir):
    """
    Create the directory structure for YOLO format dataset.

    Args:
        output_dir: Root directory for the YOLO dataset.

    Returns:
        dict: Dictionary containing paths to all split directories.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create image and label directories for each split
    splits = {}
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(output_dir, "images", split)
        labels_dir = os.path.join(output_dir, "labels", split)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        splits[split] = {'images': images_dir, 'labels': labels_dir}

    return splits


def split_dataset(image_files, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """
    Split dataset into train, validation, and test sets.

    Args:
        image_files: List of image filenames.
        train_size: Proportion of data for training.
        val_size: Proportion of data for validation.
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        dict: Dictionary with keys 'train', 'val', 'test' containing file lists.
    """
    # Calculate actual split proportions
    test_val_size = val_size + test_size
    val_proportion = val_size / test_val_size

    # First split: train vs (val + test)
    train_files, tmp_files = train_test_split(
        image_files,
        test_size=test_val_size,
        random_state=random_state,
        shuffle=True
    )

    # Second split: val vs test
    val_files, test_files = train_test_split(
        tmp_files,
        test_size=(1 - val_proportion),
        random_state=random_state,
        shuffle=True
    )

    return {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }


def process_image_and_annotation(img_path, ann_path, output_img_dir, output_lbl_dir):
    """
    Process a single image and its annotation, converting to YOLO format.

    Args:
        img_path: Path to the image file.
        ann_path: Path to the annotation XML file.
        output_img_dir: Output directory for images.
        output_lbl_dir: Output directory for labels.

    Returns:
        int: Number of annotations processed.
    """
    # Copy image
    img_name = os.path.basename(img_path)
    shutil.copy(img_path, os.path.join(output_img_dir, img_name))

    # Get image dimensions
    img_width, img_height = get_image_dimensions(img_path)

    # Parse annotation
    targets = parse_voc_annotation(ann_path)

    # Convert to YOLO format
    yolo_annotations = []
    for target in targets:
        label_name = target['label']
        class_id = CLASS_MAP[label_name]
        bbox = target['bbox']

        # Convert bbox to YOLO format
        x_center, y_center, width, height = convert_bbox_to_yolo(
            bbox, img_width, img_height
        )

        yolo_annotations.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    # Write YOLO annotation file
    base_name = os.path.splitext(img_name)[0]
    label_path = os.path.join(output_lbl_dir, base_name + ".txt")
    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))

    return len(yolo_annotations)


def convert_voc_to_yolo(images_dir, annotations_dir, output_dir,
                        train_size=0.6, val_size=0.2, test_size=0.2):
    """
    Convert Pascal VOC annotations to YOLO format.

    Args:
        images_dir: Directory containing images.
        annotations_dir: Directory containing VOC XML annotations.
        output_dir: Output directory for YOLO format dataset.
        train_size: Proportion for training set.
        val_size: Proportion for validation set.
        test_size: Proportion for test set.

    Returns:
        tuple: (dataset_dir, yaml_path) - paths to dataset and config file.
    """
    print("Converting Pascal VOC to YOLO format...")
    print(f"{'='*60}")

    # Create directory structure
    split_dirs = create_yolo_directory_structure(output_dir)

    # Get all image files
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    print(f"Total images: {len(image_files)}")

    # Split dataset
    splits = split_dataset(image_files, train_size, val_size, test_size, RANDOM_SEED)

    total_annotations = 0

    # Process each split
    for split_name, files in splits.items():
        print(f"\nProcessing {split_name} split ({len(files)} images)...")

        img_dir = split_dirs[split_name]['images']
        lbl_dir = split_dirs[split_name]['labels']

        for img_name in files:
            img_path = os.path.join(images_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            ann_path = os.path.join(annotations_dir, base_name + ".xml")

            if not os.path.exists(ann_path):
                print(f"Warning: Annotation not found for {img_name}, skipping...")
                continue

            num_annotations = process_image_and_annotation(
                img_path, ann_path, img_dir, lbl_dir
            )
            total_annotations += num_annotations

    # Create data.yaml file for YOLO
    data_yaml = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'with_mask',
            1: 'without_mask',
            2: 'mask_weared_incorrect'
        },
        'nc': 3  # number of classes
    }

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # Print summary
    print(f"\n{'='*60}")
    print("YOLO Dataset Creation Complete!")
    print(f"{'='*60}")
    print(f"Train images: {len(splits['train'])}")
    print(f"Val images: {len(splits['val'])}")
    print(f"Test images: {len(splits['test'])}")
    print(f"Total annotations: {total_annotations}")
    print(f"Dataset saved to: {output_dir}")
    print(f"Config file: {yaml_path}")
    print(f"{'='*60}")

    return output_dir, yaml_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        images_dir = sys.argv[1]
        annotations_dir = sys.argv[2]
        output_dir = sys.argv[3]

        convert_voc_to_yolo(images_dir, annotations_dir, output_dir)
    else:
        print("Usage: python converter.py <images_dir> <annotations_dir> <output_dir>")
