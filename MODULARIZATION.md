# Face Mask Detection - Module Structure

## Overview
This project has been refactored from a Jupyter notebook into modular Python components for better maintainability, reusability, and automation.

## Directory Structure

```
face-mask-detection/
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── config.py          # Configuration and constants
│   ├── download.py        # Dataset download utilities
│   ├── parser.py          # Pascal VOC annotation parsing
│   ├── converter.py       # VOC to YOLO format conversion
│   ├── train.py           # Model training
│   ├── evaluate.py        # Model evaluation
│   ├── visualize.py       # Visualization utilities
│   ├── tracking.py        # Video tracking (existing)
│   ├── utils.py           # Helper utilities
│   └── main.py            # Main pipeline orchestrator
│
├── data/                  # Dataset storage
│   ├── images/           # Original images
│   ├── annotations/      # Original VOC XML annotations
│   └── yolo_dataset/     # Converted YOLO format dataset
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── data.yaml
│
├── model/                # Model training outputs
│   └── face_mask_yolo/
│       ├── weights/
│       │   ├── best.pt
│       │   └── last.pt
│       ├── results.csv
│       └── args.yaml
│
├── runs/                 # YOLO validation/detection runs
│   └── detect/
│
├── face_mask_detection.ipynb  # Original notebook (can be archived)
├── yolo11n.pt           # Pretrained weights
├── requirements.txt
├── LICENSE
└── README.md
```

## Module Descriptions

### `config.py`
Central configuration file containing:
- Class mappings (with_mask, without_mask, mask_weared_incorrect)
- Training hyperparameters (epochs, batch size, learning rate)
- Dataset split ratios
- File paths

### `download.py`
Dataset management:
- `download_dataset()` - Download from Kaggle
- `get_dataset_paths()` - Get images/annotations paths

### `parser.py`
Pascal VOC XML parsing:
- `parse_voc_annotation()` - Parse XML files
- `get_image_dimensions()` - Get image size
- `convert_bbox_to_yolo()` - Convert bbox format

### `converter.py`
VOC to YOLO format conversion:
- `create_yolo_directory_structure()` - Setup directories
- `split_dataset()` - Train/val/test split
- `process_image_and_annotation()` - Convert single sample
- `convert_voc_to_yolo()` - Main conversion pipeline

### `train.py`
Model training:
- `load_model()` - Load YOLO model
- `train_model()` - Main training function
- `get_best_model_path()` - Get best weights path

### `evaluate.py`
Model evaluation:
- `load_model_for_evaluation()` - Load model for inference
- `evaluate_model()` - Evaluate on dataset split
- `run_inference()` - Single image inference

### `visualize.py`
Visualization functions:
- `visualize_annotations()` - Show ground truth boxes
- `predict_and_visualize()` - Show predictions vs ground truth
- `plot_class_distribution()` - Bar chart of class counts

### `utils.py`
Helper utilities:
- `ensure_dir_exists()` - Create directory
- `get_image_files()` - List images
- `get_annotation_files()` - List annotations
- `get_label_files()` - List labels
- `get_file_pairs()` - Match images with annotations

### `main.py`
Pipeline orchestrator with CLI:
- `run_full_pipeline()` - Complete workflow
- `setup_data()` - Download and prepare data
- `convert_dataset()` - Convert to YOLO format
- `train_pipeline()` - Train model
- `evaluate_pipeline()` - Evaluate model
- `visualize_predictions()` - Show test results

## Usage

### Full Pipeline
```bash
cd src
python main.py --mode full
```

### Individual Steps
```bash
# Download and prepare dataset
python main.py --mode download --force-download

# Convert to YOLO format
python main.py --mode convert

# Train model
python main.py --mode train --yaml-path ../data/yolo_dataset/data.yaml

# Evaluate model
python main.py --mode evaluate --model-path ../model/face_mask_yolo/weights/best.pt

# Visualize predictions
python main.py --mode visualize --model-path ../model/face_mask_yolo/weights/best.pt
```

### Individual Modules
```bash
# Parse annotations
python parser.py path/to/annotation.xml

# Convert dataset
python converter.py path/to/images path/to/annotations output_dir

# Visualize annotations
python visualize.py annotate image.jpg annotation.xml

# Plot class distribution
python visualize.py distribution /path/to/yolo_dataset
```

## Benefits of Modularization

1. **Reusability** - Each module can be imported and used independently
2. **Testability** - Easier to write unit tests for individual functions
3. **Maintainability** - Clear separation of concerns
4. **Scalability** - Easy to add new features or modify existing ones
5. **Automation** - CLI interface for pipeline orchestration
6. **Debugging** - Easier to identify and fix issues
7. **Documentation** - Each module has clear docstrings
8. **Version Control** - Better git history and tracking

## Data Organization

### Original Format (Notebook)
- Mixed code and output in single .ipynb file
- No clear data directory structure
- Manual step execution

### Modularized Format
- Code organized in src/ directory
- Data organized in data/ directory with clear structure
- Automated pipeline with CLI
- Reproducible and trackable workflow

## Next Steps

1. Archive the original notebook: `git mv face_mask_detection.ipynb notebooks/`
2. Run the pipeline: `python src/main.py --mode full`
3. Monitor training: Check `model/face_mask_yolo/` for results
4. Integrate with other tools (API server, monitoring dashboards, etc.)
