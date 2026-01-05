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
├── face_mask_detection.ipynb      # Main Jupyter notebook
├── README.md                       # This file
├── LICENSE                         # Project license
├── yolov11n.pt                     # Pre-trained YOLOv11n model weights
├── model/
│   └── face_mask_yolo/             # Trained model artifacts
│       ├── weights/
│       │   ├── best.pt             # Best model weights
│       │   └── last.pt             # Last epoch weights
│       ├── args.yaml               # Training configuration
│       └── results.csv             # Training metrics
└── runs/
    └── detect/
        └── val/                    # Validation results
```

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
pip install ultralytics ultralytics opencv-python matplotlib pillow beautifulsoup4 lxml pyyaml scikit-learn kagglehub torch torchvision
```

### Running the Notebook

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd face-mask-detection
   ```

2. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook face_mask_detection.ipynb
   ```

### Notebook Sections

1. **Dataset Overview**: Download and inspect the AndrewMVD Face Mask Detection dataset
2. **YOLO-Compatible Data Pipeline**: Convert Pascal VOC annotations to YOLO format
3. **Dataset Verification**: Validate the converted dataset structure
4. **Model Architecture**: Load YOLOv11n pretrained model
5. **Training**: Fine-tune the model on face mask detection task
6. **Training Results**: Visualize training metrics and progress
7. **Evaluation**: Assess performance on test set
8. **Visualization**: Display predictions vs ground truth on test images

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

Model weights and training artifacts are saved in the `model/face_mask_yolo/` directory:

- `weights/best.pt`: Best performing model (used for inference)
- `weights/last.pt`: Final epoch model
- `results.csv`: Training metrics per epoch
- `results.png`: Training visualization plots

## Usage Examples

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

## Limitations and Future Improvements

### Current Limitations

- Dataset may contain biases that affect real-world performance
- Requires further testing in diverse scenarios
- Model trained on specific hardware/conditions

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
