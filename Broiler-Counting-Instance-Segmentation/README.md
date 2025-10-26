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

The notebooks use a custom broiler chicken dataset with the following characteristics:
- **Dataset Format**: COCO JSON annotations converted to YOLO format
- **Splits**: Train, Validation, and Test sets
- **Classes**: Single class (chicken)
- **Images**: 162 test images with 845 total instances
- **Annotation Type**: Instance segmentation masks
- **Format**: YOLO format with normalized polygon coordinates
- **Note**: Dataset files are not included in this repository but can be obtained from standard COCO format poultry detection datasets

## 📁 Project Structure

This repository contains two main Jupyter notebooks implementing different approaches:

- `BroilerCounting.ipynb` - YOLOv8 Instance Segmentation training & evaluation
- `BroilerCounting(ObjectDetection).ipynb` - YOLOv8 Object Detection alternative approach

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

## 🚀 Getting Started

### Requirements

1. Install required dependencies:
```bash
pip install ultralytics opencv-python numpy matplotlib pillow pyyaml
```

2. For Google Colab usage, run the notebooks directly - all dependencies will be installed automatically.

### Running the Notebooks

1. **Instance Segmentation Approach** (`BroilerCounting.ipynb`):
   - Open the notebook in Google Colab or Jupyter
   - Follow the cells sequentially to train YOLOv8s-seg model
   - Includes data preparation, training, inference, and evaluation

2. **Object Detection Approach** (`BroilerCounting(ObjectDetection).ipynb`):
   - Open the notebook in Google Colab or Jupyter
   - Train YOLOv8n model for object detection
   - Includes training, validation, and interactive inference with image upload

### Quick Start Example

```python
from ultralytics import YOLO

# Load pre-trained or trained model
model = YOLO('yolov8s-seg.pt')  # or 'best.pt' after training

# Run inference
results = model.predict(
    source='path/to/chicken/images',
    conf=0.89,
    save=True
)

# Get count
chicken_count = len(results[0].boxes)
print(f"Detected {chicken_count} chickens")
```

## 🎨 Results

The model successfully predicts chicken instances across various scenarios:

### Key Results Achieved

The system demonstrates robust performance in:
- **Dense populations**: Accurately counts chickens in crowded conditions
- **Partial occlusion**: Handles overlapping and partially hidden chickens
- **Varied lighting**: Adapts to different illumination conditions
- **Different scales**: Detects chickens at various sizes

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

## 📓 Notebook Details

### BroilerCounting.ipynb (Instance Segmentation)
This notebook implements instance segmentation for precise chicken identification:
- **Model**: YOLOv8s-seg
- **Training**: 100 epochs with early stopping (patience=50)
- **Key Features**: 
  - COCO to YOLO conversion
  - Confidence filtering (0.85, 0.89 thresholds)
  - Comprehensive evaluation metrics
  - Batch processing and visualization

### BroilerCounting(ObjectDetection).ipynb (Object Detection)
This notebook implements object detection for chicken counting:
- **Model**: YOLOv8n (nano version for speed)
- **Training**: 50 epochs
- **Key Features**:
  - Interactive image upload and inference
  - Real-time visualization with bounding boxes
  - Custom confidence score display
  - Easy-to-use chicken count output

## 🔬 Technical Approach

The project employs two complementary approaches:

1. **Instance Segmentation**: Provides pixel-level accuracy for precise chicken boundary detection
2. **Object Detection**: Offers faster inference with bounding box localization

Both approaches achieve excellent accuracy (>97% mAP), allowing users to choose based on their specific needs and computational resources.

## 👨‍💻 Development

The project was developed using:
- **Google Colab** for GPU-accelerated training
- **YOLOv8 Ultralytics** for model architecture
- **COCO to YOLO** annotation conversion for dataset preparation
- **Custom confidence filtering** for optimal precision-recall balance

## 📄 License

This project is available for educational and research purposes.

---

**Note**: This project demonstrates the application of state-of-the-art computer vision techniques to solve real-world problems in poultry farming, combining deep learning with practical agricultural needs.

**Keywords**: YOLOv8, Computer Vision, Deep Learning, Instance Segmentation, Object Detection, Poultry Counting, Agricultural AI, Automated Counting System

