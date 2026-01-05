"""Helper utilities for the face mask detection project."""

import os


def ensure_dir_exists(path):
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def get_image_files(directory, extensions=('.png', '.jpg', '.jpeg')):
    """
    Get list of image files in a directory.

    Args:
        directory: Path to directory.
        extensions: Tuple of file extensions to look for.

    Returns:
        list: Sorted list of image filenames.
    """
    if not os.path.exists(directory):
        return []

    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(extensions)
    ])


def get_annotation_files(directory):
    """
    Get list of annotation XML files in a directory.

    Args:
        directory: Path to directory.

    Returns:
        list: Sorted list of annotation filenames.
    """
    if not os.path.exists(directory):
        return []

    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith('.xml')
    ])


def get_label_files(directory):
    """
    Get list of YOLO label TXT files in a directory.

    Args:
        directory: Path to directory.

    Returns:
        list: Sorted list of label filenames.
    """
    if not os.path.exists(directory):
        return []

    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith('.txt')
    ])


def get_file_pairs(images_dir, annotations_dir):
    """
    Get pairs of image and annotation files.

    Args:
        images_dir: Directory containing images.
        annotations_dir: Directory containing annotations.

    Returns:
        list: List of tuples (image_path, annotation_path) for matched pairs.
    """
    image_files = get_image_files(images_dir)
    annotation_files = get_annotation_files(annotations_dir)

    pairs = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        ann_file = base_name + '.xml'

        if ann_file in annotation_files:
            img_path = os.path.join(images_dir, img_file)
            ann_path = os.path.join(annotations_dir, ann_file)
            pairs.append((img_path, ann_path))

    return pairs
