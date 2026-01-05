"""Module for visualization of face mask detection results."""

import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

from parser import parse_voc_annotation
from config import REVERSE_CLASS_MAP, CLASS_MAP


def visualize_annotations(image_path, annotation_path, save_path=None):
    """
    Visualize ground truth annotations on an image.

    Args:
        image_path: Path to the image file.
        annotation_path: Path to the XML annotation file.
        save_path: Optional path to save the visualization.
    """
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Parse annotations
    targets = parse_voc_annotation(annotation_path)

    # Create figure
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img_rgb)

    # Draw bounding boxes
    for target in targets:
        xmin, ymin, xmax, ymax = target["bbox"]
        label = target["label"]

        width = xmax - xmin
        height = ymax - ymin

        rect = patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rect)

        ax.text(
            xmin,
            ymin - 5,
            label,
            color="red",
            fontsize=10,
            backgroundcolor="white"
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Ground Truth Annotations', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()


def predict_and_visualize(image_path, model, annotation_path=None,
                         confidence_threshold=0.5, save_path=None):
    """
    Run YOLO inference on an image and visualize predictions with optional ground truth.

    Args:
        image_path: Path to the image file.
        model: YOLO model for inference.
        annotation_path: Optional path to ground truth XML annotation.
        confidence_threshold: Confidence threshold for filtering detections.
        save_path: Optional path to save the visualization.
    """
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run inference
    if hasattr(model, 'predict'):
        results = model.predict(source=image_path, conf=confidence_threshold, verbose=False)[0]
    else:
        from evaluate import run_inference
        results = run_inference(model, image_path, confidence_threshold)

    # Load ground truth if available
    gt_boxes = []
    gt_labels = []

    if annotation_path and os.path.exists(annotation_path):
        targets = parse_voc_annotation(annotation_path)
        for target in targets:
            gt_boxes.append(target['bbox'])
            gt_labels.append(target['label'])

    num_plots = 2 if gt_boxes else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(10 * num_plots, 8))

    if num_plots == 1:
        axes = [axes]

    class_names = list(REVERSE_CLASS_MAP.values())

    # Plot predictions
    axes[0].imshow(img_rgb)
    axes[0].set_title('Predictions', fontsize=16, fontweight='bold')

    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls in zip(boxes, scores, classes):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin

            rect = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor="green",
                facecolor="none"
            )
            axes[0].add_patch(rect)

            label_text = f"{class_names[cls]}: {score:.2f}"
            axes[0].text(
                xmin,
                ymin - 5,
                label_text,
                color="green",
                fontsize=10,
                backgroundcolor="white"
            )

    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Plot ground truth if available
    if gt_boxes:
        axes[1].imshow(img_rgb)
        axes[1].set_title('Ground Truth', fontsize=16, fontweight='bold')

        for box, label in zip(gt_boxes, gt_labels):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin

            rect = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )
            axes[1].add_patch(rect)

            axes[1].text(
                xmin,
                ymin - 5,
                label,
                color="red",
                fontsize=10,
                backgroundcolor="white"
            )

        axes[1].set_xticks([])
        axes[1].set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()


def plot_class_distribution(yolo_dataset_dir, save_path=None):
    """
    Plot the distribution of bounding boxes per class in the dataset.

    Args:
        yolo_dataset_dir: Path to the YOLO format dataset directory.
        save_path: Optional path to save the plot.
    """
    # Count boxes per class
    class_counts = defaultdict(int)

    # Get all label files from train, val, and test splits
    label_dirs = [
        os.path.join(yolo_dataset_dir, "labels", "train"),
        os.path.join(yolo_dataset_dir, "labels", "val"),
        os.path.join(yolo_dataset_dir, "labels", "test")
    ]

    for label_dir in label_dirs:
        if not os.path.exists(label_dir):
            continue

        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                label_path = os.path.join(label_dir, label_file)
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            class_counts[class_id] += 1

    # Prepare data for plotting
    class_names = [REVERSE_CLASS_MAP[i] for i in sorted(class_counts.keys())]
    counts = [class_counts[i] for i in sorted(class_counts.keys())]

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(class_names, counts, color=['#2ecc71', '#e74c3c', '#f39c12'])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Number of Boxes', fontsize=12, fontweight='bold')
    ax.set_xlabel('Class Label', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Bounding Boxes per Class', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Class distribution plot saved to: {save_path}")

    plt.show()

    # Print summary
    print(f"\n{'='*60}")
    print("Class Distribution Summary:")
    print(f"{'='*60}")
    for name, count in zip(class_names, counts):
        percentage = (count / sum(counts)) * 100
        print(f"{name}: {count} boxes ({percentage:.1f}%)")
    print(f"{'='*60}")
    print(f"Total boxes: {sum(counts)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        mode = sys.argv[1]

        if mode == "annotate":
            image_path = sys.argv[2]
            annotation_path = sys.argv[3]
            save_path = sys.argv[4] if len(sys.argv) > 4 else None
            visualize_annotations(image_path, annotation_path, save_path)

        elif mode == "distribution":
            dataset_dir = sys.argv[2]
            save_path = sys.argv[3] if len(sys.argv) > 3 else None
            plot_class_distribution(dataset_dir, save_path)
    else:
        print("Usage:")
        print("  Visualize annotations: python visualize.py annotate <image> <annotation> [save_path]")
        print("  Plot distribution: python visualize.py distribution <dataset_dir> [save_path]")
