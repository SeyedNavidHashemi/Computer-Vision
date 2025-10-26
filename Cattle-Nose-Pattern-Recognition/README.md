# Deep Cow Identification System

A deep learning-based cattle identification system using nose pattern recognition. This project implements multiple state-of-the-art architectures to classify cows as known or unknown and identify specific cow IDs through biometric pattern analysis.

## 🎯 Overview

This system uses computer vision and deep learning to identify individual cows by analyzing their unique nose patterns. The solution can distinguish between registered (known) and unregistered (unknown) cattle and accurately predict the ID of known animals.

## 🚀 Features

- **Multi-Architecture Support**: Implements EfficientNet-B5, ResNet50, and Custom CNN
- **Known/Unknown Classification**: Determines whether a cow is in the database
- **ID Prediction**: Identifies specific cow IDs for known animals
- **High Accuracy**: Achieves 89-92% accuracy on known/unknown classification and 95%+ accuracy on ID prediction
- **Web Interface**: Persian web UI for easy testing
- **Multiple Training Strategies**: Contrastive learning, triplet loss, and data augmentation

## 📊 Model Performance

### UsingEfficientNet-B5 (Best Results)
- **Known/Unknown Classification**: 91% Accuracy
- **ID Prediction Accuracy**: 95.9%
- **ROC AUC**: 0.96
- **F1 Score**: 0.91

### UsingResNet50
- **Known/Unknown Classification**: 92% Accuracy
- **ID Prediction Accuracy**: 95%+
- **ROC AUC**: 0.97

## 📁 Project Structure

```
CowDetection/
├── train/                    # Training images (3383 images)
│   └── cattle_*/             # Individual cow folders
├── val/                      # Validation images (603 images)
│   └── cattle_*/             # Individual cow folders
├── UI/                       # Web interface
│   ├── index.html            # Persian UI for cow identification
│   └── style.css             # Styling
├── CustomeCNN.ipynb          # Custom CNN with contrastive learning
├── UsingEfficientNet50(35epochs+data augmentation).ipynb  # EfficientNet-B5 implementation
├── UsingEfficientNet5B.ipynb # EfficientNet-B5 with triplet loss
└── UsingResNet50.ipynb       # ResNet50 implementation
```

## 🔧 Technical Details

### Architecture Choices

#### 1. EfficientNet-B5 (Recommended)
- **Backbone**: EfficientNet-B5 (ImageNet pre-trained)
- **Loss Function**: Triplet Loss (margin=1.5)
- **Embedding Dimension**: 128
- **Training Strategy**: 
  - Freeze backbone for 10 epochs
  - Fine-tune entire network for remaining epochs
  - Data augmentation (horizontal flip, rotation, color jitter)

#### 2. ResNet50
- **Backbone**: ResNet50 (ImageNet pre-trained)
- **Loss Function**: Contrastive Loss
- **Embedding Dimension**: 128
- **Training Strategy**: Full fine-tuning with augmentation

#### 3. Custom CNN
- **Architecture**: Custom convolutional layers
- **Loss Function**: Contrastive Loss
- **Training Strategy**: End-to-end training

### Training Approach

1. **Data Preparation**:
   - Split training data into known (66%) and unknown (33%) classes
   - Generate triplet pairs (anchor, positive, negative)
   - Apply data augmentation

2. **Loss Functions**:
   - **Triplet Loss**: `max(0, d(anchor, positive) - d(anchor, negative) + margin)`
   - **Contrastive Loss**: Minimizes distance for similar pairs, maximizes for dissimilar pairs

3. **Inference**:
   - Extract embeddings using trained encoder
   - Use Nearest Neighbors to find closest match
   - Apply distance threshold for known/unknown classification
   - Predict ID based on nearest neighbor

## 🛠️ Installation

### Prerequisites
```bash
pip install torch torchvision
pip install torchvision
pip install albumentations
pip install opencv-python
pip install scikit-learn
pip install numpy
pip install tqdm
pip install matplotlib seaborn
```

### Running the Notebooks

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Choose a model**:
   - `UsingEfficientNet50(35epochs+data augmentation).ipynb` - Best performing model
   - `UsingResNet50.ipynb` - Good alternative
   - `CustomeCNN.ipynb` - Lighter model

3. **Update paths** (if not using Google Colab):
   - Modify `train_dir` and `val_dir` paths in the notebooks
   - Ensure your data follows the structure: `train/cattle_ID/images.jpg`

## 📈 Usage

### Training

1. Prepare your dataset following the structure:
   ```
   train/
   ├── cattle_1000/
   │   ├── image1.jpg
   │   └── image2.jpg
   ├── cattle_1001/
   └── ...
   ```

2. Run the training notebook:
   - The system will automatically create train/val splits
   - Training checkpoints will be saved automatically

### Inference

1. **Using Web UI**:
   - Open `UI/index.html` in a browser
   - Upload a cow nose image
   - Get instant identification results

2. **Using Python**:
   ```python
   # Load trained model
   model = Encoder(embedding_dim=128)
   model.load_state_dict(torch.load('checkpoint.pth'))
   
   # Extract embeddings and classify
   embeddings = extract_embeddings(model, image_paths, transform)
   predictions = classify_cow(embeddings, knn_model, threshold)
   ```

## 📝 Dataset

- **Training Images**: 3,383 images across multiple cow IDs
- **Validation Images**: 603 images
- **Image Size**: 128x128 or 224x224 (depending on model)
- **Classes**: Individual cow IDs (e.g., cattle_1000, cattle_2000)

## 🎓 Key Concepts

### Known vs Unknown Classification
- **Known**: Cows present in the training database
- **Unknown**: New cows not seen during training
- Uses distance-based thresholding for classification

### ID Prediction
- Only applies to cows classified as "Known"
- Uses k-Nearest Neighbors (k=1) for identification
- Predicts the cow ID based on the closest embedding

### Contrastive Learning
- Learns to bring similar images (same cow) closer
- Pushes dissimilar images (different cows) apart
- Enables the model to learn distinctive cow features

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| EfficientNet-B5 | 91.0% | 91.2% | 90.4% | 0.91 |
| ResNet50 | 92.0% | 93.1% | 92.8% | 0.93 |
| Custom CNN | 71.0% | 70.7% | 73.1% | 0.72 |

## 🔬 Technical Highlights

- **Transfer Learning**: Leverages ImageNet pre-trained weights
- **Data Augmentation**: Horizontal flip, rotation, brightness/contrast adjustment
- **Checkpoint Management**: Automatic model checkpointing during training
- **Distance Metrics**: Cosine similarity and L2 distance
- **Threshold Optimization**: F1-score optimized thresholds for best performance

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for educational and research purposes.

## 👨‍💻 Author

Developed as part of an internship project on automated livestock management.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- TensorFlow team for EfficientNet architecture
- Albumentations for data augmentation
- Google Colab for computing resources

---

**Note**: This system is designed for cattle identification in agricultural and livestock management contexts. The accuracy may vary based on image quality, lighting conditions, and cow pose.

