# Face Mask Detection with YOLOv11

A modern object detection project that detects and classifies face mask usage in images using **YOLOv11** and the **AndrewMVD Face Mask Detection** dataset.

## Objective

Detect and localize face mask usage in images, classifying faces into three categories:

- **with_mask**: Person wearing a mask correctly
- **without_mask**: Person not wearing a mask
- **mask_weared_incorrect**: Person wearing a mask incorrectly

## Dataset

This project uses the **Face Mask Detection** dataset by AndrewMVD, which contains:

- **Source**: Kaggle - [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- **Format**: Originally in Pascal VOC (XML) format
- **Content**: Images of people annotated with face-level bounding boxes indicating mask usage
- **Multi-face**: Each image may contain multiple faces with different mask states

The annotations are converted from **Pascal VOC** to **YOLO format** for maximum compatibility with Ultralytics YOLOv11.

## Approach

1. **Data Preparation**: Download and parse Pascal VOC annotations
2. **Format Conversion**: Convert VOC XML annotations to YOLO format (normalized center coordinates)
3. **Dataset Splitting**: Divide data into train (60%), validation (20%), and test (20%) sets
4. **Model Selection**: Use Ultralytics YOLOv11n (nano) pretrained on COCO
5. **Fine-tuning**: Train on face mask detection task with SGD optimizer
6. **Evaluation**: Assess performance using standard metrics (mAP, precision, recall)
7. **Visualization**: Demonstrate predictions on test images

## Project Structure

```
face-mask-detection/
├── src/                           # Modularized source code
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Configuration and hyperparameters
│   ├── download.py               # Dataset download from Kaggle
│   ├── parser.py                 # Pascal VOC annotation parsing
│   ├── converter.py              # VOC to YOLO format conversion
│   ├── train.py                  # Model training pipeline
│   ├── evaluate.py               # Model evaluation and inference
│   ├── visualize.py              # Visualization utilities
│   ├── utils.py                  # Helper utilities
│   ├── main.py                   # Pipeline orchestrator with CLI
│   └── tracking.py               # Video object tracking
├── data/                         # Dataset storage
│   ├── images/                   # Original images
│   ├── annotations/              # Original VOC XML annotations
│   └── yolo_dataset/             # Converted YOLO format dataset
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── data.yaml
├── model/                        # Model training outputs
│   └── face_mask_yolo/
│       ├── weights/
│       │   ├── best.pt
│       │   └── last.pt
│       ├── args.yaml
│       └── results.csv
├── runs/                         # YOLO validation/detection runs
├── MODULARIZATION.md             # Detailed modularization guide
├── README.md                      # This file
├── LICENSE
└── requirements.txt
```

## Architecture

This project uses **modular Python components** for better maintainability, reusability, and automation. All functionality is implemented as standalone modules that can be used together via the CLI or imported individually in Python code. See [MODULARIZATION.md](MODULARIZATION.md) for detailed documentation.

### Core Modules

- **config.py**: Centralized configuration with hyperparameters and class mappings
- **download.py**: Automated dataset download from Kaggle
- **parser.py**: Pascal VOC XML annotation parsing utilities
- **converter.py**: VOC to YOLO format conversion pipeline
- **train.py**: Model training with YOLO support
- **evaluate.py**: Model evaluation and inference capabilities
- **visualize.py**: Visualization tools for annotations and predictions
- **utils.py**: Reusable helper functions
- **main.py**: CLI orchestrator for the complete pipeline

## Model Architecture: YOLOv11

**YOLOv11** is the latest evolution in the YOLO (You Only Look Once) family of real-time object detectors. Key advantages:

- **Speed**: Single-stage detector optimized for real-time inference
- **Accuracy**: State-of-the-art performance with improved architecture
- **Simplicity**: Easy-to-use API through Ultralytics
- **Flexibility**: Multiple model sizes (nano, small, medium, large, xlarge)

Unlike two-stage detectors like Faster R-CNN, YOLO predicts bounding boxes and class probabilities directly from full images in one evaluation, making it much faster while maintaining excellent accuracy.

This project uses **YOLOv11n** (nano variant) for fast inference on standard hardware.

## Quick Start

### Prerequisites

```bash
python --version # Must be >= 3.10
mamba create -n face-mask-detection python=3.10
mamba activate face-mask-detection
pip install -r requirements.txt
```

### Running the Full Pipeline

```bash
cd src
python main.py --mode full
```

This will:
1. Download the dataset from Kaggle
2. Convert annotations to YOLO format
3. Display class distribution statistics
4. Evaluate the pre-trained model
5. Visualize predictions on test images

### Individual Pipeline Modes

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

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch Size | 16 |
| Image Size | 640×640 |
| Early Stopping Patience | 7 epochs |
| Optimizer | SGD |
| Learning Rate | 0.01 |
| Momentum | 0.937 |
| Weight Decay | 0.0005 |

## Class Distribution

The dataset contains three classes with the following characteristics:

- **with_mask**: Correct mask usage
- **without_mask**: No face protection
- **mask_weared_incorrect**: Misplaced or incorrect mask

## Evaluation Metrics

The trained model is evaluated using standard object detection metrics:

- **mAP50**: Mean Average Precision at IoU threshold 0.50
- **mAP50-95**: Mean Average Precision averaged across IoU thresholds 0.50-0.95
- **Precision**: Proportion of correct positive predictions
- **Recall**: Proportion of actual positives correctly identified
- **Per-class AP50**: Average precision for each mask class

## Results

### Test Set Performance

The trained YOLOv11n model achieves the following metrics on the test set:

| Metric | Value |
|--------|-------|
| **mAP50** | 0.7050 |
| **mAP50-95** | 0.4524 |
| **Precision** | 0.7275 |
| **Recall** | 0.6606 |

### Per-Class Performance

| Class | AP50 |
|-------|------|
| with_mask | 0.9297 ⭐ (excellent) |
| without_mask | 0.7860 ✓ (good) |
| mask_weared_incorrect | 0.3994 ⚠️ (needs improvement) |

Model weights and training artifacts are saved in the `model/face_mask_yolo/` directory:

- `weights/best.pt`: Best performing model (used for inference)
- `weights/last.pt`: Final epoch model
- `results.csv`: Training metrics per epoch
- Results visualization plots

## Usage Examples

### Using Individual Modules

All modules can be imported and used independently in your Python code:

```python
from src.download import download_dataset, get_dataset_paths
from src.converter import convert_voc_to_yolo
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import predict_and_visualize

# Download dataset
dataset_path = download_dataset()
images_dir, annotations_dir = get_dataset_paths(dataset_path)

# Convert to YOLO format
dataset_dir, yaml_path = convert_voc_to_yolo(images_dir, annotations_dir, 'data/yolo_dataset')

# Train model
model = train_model(yaml_path)

# Evaluate
results = evaluate_model('model/face_mask_yolo/weights/best.pt', yaml_path)
```

### Loading and Using the Model

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('model/face_mask_yolo/weights/best.pt')

# Run inference on an image
results = model.predict(source='path/to/image.jpg', conf=0.5)

# Results contain bounding boxes, confidence scores, and class predictions
for result in results:
    for box in result.boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf:.2f}")
```

### Batch Processing

```python
# Predict on multiple images
results = model.predict(source='path/to/images/', conf=0.5, batch=16)
```

## Object Tracking with Tracker

The project includes a real-time object tracking script (`src/tracking.py`) that extends detection capabilities by tracking individual masks across video frames. This is useful for:

- **Person Identification**: Track the same person across multiple frames
- **Compliance Monitoring**: Monitor mask compliance over time
- **Crowd Analysis**: Count unique individuals and their mask status
- **Video Analysis**: Process video streams or webcam feeds

### Running the Tracker

#### Webcam Stream (Default)

```bash
cd src
python3 tracking.py
```

#### With ByteTrack (Default)

```bash
python3 tracking.py --bytetrack
```

#### With BotSORT Tracker

```bash
python3 tracking.py --botsort
```

#### Video File Input

```bash
python3 tracking.py --source path/to/video.mp4
```

### Tracker Options

The tracking script supports various command-line arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bytetrack` | flag | - | Use ByteTrack tracker (default) |
| `--botsort` | flag | - | Use BotSORT tracker |
| `--source` | str | `0` | Video source (0 for webcam, path for video file, URL for stream) |
| `--conf` | float | `0.5` | Confidence threshold for detections |
| `--iou` | float | `0.5` | IoU threshold for NMS |
| `--device` | str | None | Device to run on (cuda, cpu, 0, 1, etc.) |
| `--no-show` | flag | False | Disable displaying the video output |
| `--save` | flag | False | Save the output video |
| `--model-path` | str | `../model/face_mask_yolo/weights/best.pt` | Path to model weights |

### Tracker Comparison

- **ByteTrack**: Fast, lightweight tracker suitable for real-time applications
- **BotSORT**: More robust tracker with better handling of occlusions and crowded scenes

### Example Usage

Track masks from webcam with BotSORT and save output:

```bash
python3 tracking.py --botsort --save --device cuda
```

Track from a video file with ByteTrack at custom confidence threshold:

```bash
python3 tracking.py --source video.mp4 --bytetrack --conf 0.6
```

Track from a USB camera (device 1) without displaying output:

```bash
python3 tracking.py --source 1 --no-show
```

## Architecture & Design

### Modularization Benefits

The refactored architecture provides:

✅ **Separation of Concerns** - Each module handles a specific task
✅ **Reusability** - Import and use modules independently
✅ **Testability** - Easier to write unit tests
✅ **Maintainability** - Clear code organization
✅ **Scalability** - Easy to extend functionality
✅ **Automation** - CLI interface for pipeline orchestration

For detailed information, see [MODULARIZATION.md](MODULARIZATION.md).

## Limitations and Future Improvements

### Current Limitations

- Dataset may contain biases that affect real-world performance
- Model trained on specific hardware/conditions
- `mask_weared_incorrect` class has lower accuracy (needs more training data)
- Requires further testing in diverse scenarios

### Future Enhancements

- Expand dataset with more diverse examples and real-world scenarios
- Experiment with different YOLO model sizes:
  - `yolov11n`: Nano (edge devices, fastest)
  - `yolov11m`: Medium (balanced)
  - `yolov11l`: Large (highest accuracy)
  - `yolov11x`: Extra-large (maximum accuracy)
- Add confusion matrices and detailed per-class analysis
- Implement real-time video detection
- Deploy using ONNX or TensorRT for optimized inference
- Fine-tune hyperparameters for specific use cases
- Test on different mask types and lighting conditions
- Integrate with applications (security systems, health monitoring)

## Dependencies

Key libraries used in this project:

- **ultralytics**: YOLOv11 implementation and training framework
- **torch/torchvision**: Deep learning framework
- **opencv-python**: Computer vision utilities
- **matplotlib**: Data visualization
- **pillow**: Image processing
- **beautifulsoup4**: XML parsing for Pascal VOC annotations
- **scikit-learn**: Train/test splitting
- **kagglehub**: Dataset downloading

## References

- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)
- [AndrewMVD Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- [YOLO: You Only Look Once Papers](https://pjreddie.com/darknet/yolo/)

## License

This project follows MIT license and is intended for educational/portfolio purposes. See [LICENSE](./LICENSE).

## Conclusion

This project successfully demonstrates face mask detection using YOLOv11, achieving strong performance with a modern, efficient architecture. The workflow showcases:

- Modern object detection techniques
- Practical data preprocessing and conversion
- Effective model training and evaluation
- Real-time inference capabilities

The combination of YOLOv11's speed and accuracy makes it suitable for production face mask detection systems in surveillance, access control, and health monitoring applications.
