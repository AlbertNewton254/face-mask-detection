"""Module for parsing Pascal VOC XML annotations."""

import os
from bs4 import BeautifulSoup
from PIL import Image


def parse_voc_annotation(annotation_path):
    """
    Parse a Pascal VOC XML annotation file.

    Args:
        annotation_path: Path to the XML annotation file.

    Returns:
        list: List of dictionaries containing label and bbox information.
              Each dict has keys: 'label' and 'bbox' (as [xmin, ymin, xmax, ymax]).
    """
    with open(annotation_path, 'r') as f:
        soup = BeautifulSoup(f, "xml")

    objects = soup.find_all("object")
    targets = []

    for obj in objects:
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        targets.append({
            "label": label,
            "bbox": [xmin, ymin, xmax, ymax]
        })

    return targets


def get_image_dimensions(image_path):
    """
    Get the dimensions of an image.

    Args:
        image_path: Path to the image file.

    Returns:
        tuple: (width, height) of the image.
    """
    img = Image.open(image_path)
    return img.size


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert Pascal VOC bbox format to YOLO format.

    Args:
        bbox: Bounding box in [xmin, ymin, xmax, ymax] format.
        img_width: Width of the image.
        img_height: Height of the image.

    Returns:
        tuple: (x_center, y_center, width, height) in normalized coordinates (0-1).
    """
    xmin, ymin, xmax, ymax = bbox

    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height

    return x_center, y_center, width, height


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        annotation_file = sys.argv[1]
        targets = parse_voc_annotation(annotation_file)
        print(f"Parsed {len(targets)} objects:")
        for target in targets:
            print(f"  {target}")
