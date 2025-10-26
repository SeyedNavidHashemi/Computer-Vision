# Image to Sketch Converter

A computer vision application that converts images and videos into pencil sketches using OpenCV.

## Overview

This project implements a 5-step image processing algorithm to create realistic pencil sketch effects from photographs and video streams in real-time.

## Features

- Static image to sketch conversion
- Real-time video processing
- Webcam integration with live preview
- FPS monitoring

## Algorithm

The sketch conversion follows these steps:
1. Convert image to grayscale
2. Invert the grayscale image
3. Apply Gaussian blur (21x21 kernel)
4. Invert the blurred image
5. Divide original grayscale by inverted blur with scaling

**Mathematical Formula:**
```
sketch = gray_image / (255 - GaussianBlur(255 - gray_image)) * 256.0
```

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib

## Usage

Open the Jupyter notebook `Image_to_Sketch.ipynb` and run the cells to convert images or process video files.
