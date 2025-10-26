# Broiler Chicken Counting with YOLOv8

A deep learning-based computer vision system for automated counting and segmentation of broiler chickens in farm environments using YOLOv8 instance segmentation and object detection models.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Usage](#usage)
- [Results](#results)
- [Key Achievements](#key-achievements)

## 🎯 Overview

This project implements an automated system for counting broiler chickens in farm images using state-of-the-art YOLOv8 models. The system can detect and count chickens with high accuracy, providing valuable insights for poultry farm management. It leverages both object detection and instance segmentation approaches to achieve robust performance across various farm conditions.

## ✨ Features

- **Dual Approach**: Implements both object detection and instance segmentation for maximum flexibility
- **High Accuracy**: Achieves 97%+ mAP50 on test dataset
- **Instance Segmentation**: Provides pixel-level segmentation masks for precise chicken identification
- **Confidence Filtering**: Adjustable confidence thresholds (0.85, 0.89) for different precision requirements
- **Batch Processing**: Efficient processing of multiple images and frames
- **Detailed Evaluation**: Comprehensive metrics including precision, recall, F1-score, and confusion matrices
- **Visualization Tools**: Rich visualization of predictions with segmentation masks and bounding boxes

## 🛠 Technologies

- **YOLOv8**: State-of-the-art object detection and segmentation model
  - YOLOv8s-seg (Instance Segmentation)
  - YOLOv8n (Object Detection)
- **Python 3.11-3.12**
- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLO model training and inference
- **OpenCV**: Image processing and visualization
- **NumPy**: Numerical operations
- **Matplotlib**: Plotting and visualization
- **Google Colab**: Cloud-based GPU training

## 📊 Dataset

- **Dataset Format**: COCO JSON annotations converted to YOLO format
- **Splits**: Train, Validation, and Test sets
- **Classes**: Single class (chicken)
- **Images**: 162 test images with 845 total instances
- **Annotation Type**: Instance segmentation masks
- **Format**: YOLO format with normalized polygon coordinates

## 📁 Project Structure

```
BroilerCounting/
├── BroilerCounting.ipynb                     # YOLOv8 Instance Segmentation training & evaluation
├── BroilerCounting(ObjectDetection).ipynb    # YOLOv8 Object Detection alternative approach
├── download_videos_from_youtube.py           # Video extraction utility
├── chicken_frames/                          # Extracted frames from videos
├── chicken_frames_cropped/                  # Cropped frames for processing
├── chicken_videos/                          # Source video files
├── results_for_counting_with_YOLO-v8/       # Counting model results
│   ├── output_filtered_masks0.85/          # Results with 0.85 confidence threshold
│   ├── output_filtered_masks0.89/          # Results with 0.89 confidence threshold
│   └── test_evaluation_run/                # Evaluation metrics and visualizations
└── results_for_growth_curve_with_YOLO-v8/   # Growth monitoring results
    ├── general_prediction_on_the_test_set/
    └── test_evaluation_run/
```

## 📈 Model Performance

### Instance Segmentation Model (YOLOv8s-seg)

**Training Configuration:**
- Model: YOLOv8s-seg
- Epochs: 100
- Image Size: 640x640
- Batch Size: 16
- Patience: 50 (Early Stopping)

**Test Results:**
```
Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95)
all        162        845       0.972      0.976      0.980      0.961      0.972      0.976      0.980      0.969
```

**Key Metrics:**
- **Precision (Box)**: 97.2%
- **Recall (Box)**: 97.6%
- **mAP@0.5 (Box)**: 98.0%
- **mAP@0.5-0.95 (Box)**: 96.1%
- **Precision (Mask)**: 97.2%
- **Recall (Mask)**: 97.6%
- **mAP@0.5 (Mask)**: 98.0%
- **mAP@0.5-0.95 (Mask)**: 96.9%

### Object Detection Model (YOLOv8n)

**Training Configuration:**
- Model: YOLOv8n
- Epochs: 50
- Image Size: 640x640

**Validation Results:**
```
Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
all         14        103       0.969      0.961      0.989      0.869
```

**Key Metrics:**
- **Precision**: 96.9%
- **Recall**: 96.1%
- **mAP@0.5**: 98.9%
- **mAP@0.5-0.95**: 86.9%

## 🚀 Usage

### Training

1. **For Instance Segmentation:**

```python
from ultralytics import YOLO

# Load YOLOv8s-seg model
model = YOLO('yolov8s-seg.pt')

# Train the model
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=50
)
```

2. **For Object Detection:**

```python
from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(data='data.yaml', epochs=50, imgsz=640)
```

### Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('best.pt')

# Run inference
results = model.predict(
    source='path/to/images',
    conf=0.89,  # Confidence threshold
    save=True,
    show_boxes=True,
    show_labels=True
)

# Count detected chickens
chicken_count = len(results[0].boxes)
print(f"Chickens detected: {chicken_count}")
```

### Evaluation

```python
# Evaluate on test set
results = model.val(
    data='data.yaml',
    split='test',
    imgsz=640,
    batch=16,
    plots=True
)
```

## 🎨 Results

The model successfully predicts chicken instances across various scenarios:

### Example Predictions

The system demonstrates robust performance in:
- **Dense populations**: Accurately counts chickens in crowded conditions
- **Partial occlusion**: Handles overlapping and partially hidden chickens
- **Varied lighting**: Adapts to different illumination conditions
- **Different scales**: Detects chickens at various sizes

#### Sample Detection Results

![Sample Detection 1](results_for_counting_with_YOLO-v8/output_filtered_masks0.89/000000000018_jpg.rf.48cbc904f3f968230adf1edea8140d39.jpg)

![Sample Detection 2](results_for_counting_with_YOLO-v8/output_filtered_masks0.89/000000000056_jpg.rf.7298a461ff68eab650ae921c6c30d507.jpg)

![Sample Detection 3](results_for_counting_with_YOLO-v8/output_filtered_masks0.89/000000000089_jpg.rf.70db9f2e9176f4bdc6ecee5199bd2db2.jpg)

![Sample Detection 4](results_for_counting_with_YOLO-v8/output_filtered_masks0.89/000000000095_jpg.rf.c7ef0118ac922e29fee82b9e5d5a5375.jpg)

![Sample Detection 5](results_for_counting_with_YOLO-v8/output_filtered_masks0.89/000000000976_jpg.rf.68da79ca39e3e23df627653ed8d22e53.jpg)

### Evaluation Metrics Visualization

Results include comprehensive evaluation metrics and visualizations:

#### Confusion Matrix
![Confusion Matrix](results_for_counting_with_YOLO-v8/test_evaluation_run/confusion_matrix_normalized.png)

#### Precision-Recall Curves
![Box Precision-Recall Curve](results_for_counting_with_YOLO-v8/test_evaluation_run/BoxPR_curve.png)

![Mask Precision-Recall Curve](results_for_counting_with_YOLO-v8/test_evaluation_run/MaskPR_curve.png)

#### F1 Score Curves
![Box F1 Curve](results_for_counting_with_YOLO-v8/test_evaluation_run/BoxF1_curve.png)

![Mask F1 Curve](results_for_counting_with_YOLO-v8/test_evaluation_run/MaskF1_curve.png)

#### Validation Batch Comparisons

**Ground Truth vs Predictions - Batch 0:**
![Validation Batch 0 Labels](results_for_counting_with_YOLO-v8/test_evaluation_run/val_batch0_labels.jpg)

![Validation Batch 0 Predictions](results_for_counting_with_YOLO-v8/test_evaluation_run/val_batch0_pred.jpg)

**Ground Truth vs Predictions - Batch 1:**
![Validation Batch 1 Labels](results_for_counting_with_YOLO-v8/test_evaluation_run/val_batch1_labels.jpg)

![Validation Batch 1 Predictions](results_for_counting_with_YOLO-v8/test_evaluation_run/val_batch1_pred.jpg)

**Ground Truth vs Predictions - Batch 2:**
![Validation Batch 2 Labels](results_for_counting_with_YOLO-v8/test_evaluation_run/val_batch2_labels.jpg)

![Validation Batch 2 Predictions](results_for_counting_with_YOLO-v8/test_evaluation_run/val_batch2_pred.jpg)

### Output Formats

- **Bounding Boxes**: Rectangular coordinates for object detection
- **Segmentation Masks**: Pixel-level masks for precise localization
- **Confidence Scores**: Probabilistic predictions for each detection
- **Counts**: Total number of chickens detected per image

## 🏆 Key Achievements

1. **High Accuracy**: Achieved 98% mAP@0.5 on the test dataset
2. **Robust Segmentation**: Precise mask generation with 96.9% mask mAP
3. **Dual Implementation**: Both detection and segmentation approaches implemented
4. **Production Ready**: Fast inference with ~15-20ms per image on GPU
5. **Scalable Solution**: Efficient batch processing for real-world applications
6. **Comprehensive Evaluation**: Detailed metrics and visualizations for analysis

## 📝 Applications

- **Poultry Farm Management**: Automated counting for inventory tracking
- **Growth Monitoring**: Track chicken population over time
- **Health Surveillance**: Monitor distribution patterns
- **Feed Optimization**: Optimize feed distribution based on accurate counts
- **Biosecurity**: Ensure proper population density management

## 🔧 Requirements

```
ultralytics>=8.3.0
torch>=1.8.0
torchvision>=0.9.0
opencv-python>=4.6.0
numpy>=1.23.0
matplotlib>=3.3.0
pillow>=7.1.2
pyyaml>=5.3.1
```

## 📄 License

This project is developed for research and educational purposes.

## 👨‍💻 Development

The project was developed using:
- **Google Colab** for GPU-accelerated training
- **YOLOv8 Ultralytics** for model architecture
- **COCO to YOLO** annotation conversion for dataset preparation
- **Custom confidence filtering** for optimal precision-recall balance

---

**Note**: This project demonstrates the application of state-of-the-art computer vision techniques to solve real-world problems in poultry farming, combining deep learning with practical agricultural needs.


