# Image Alignment

A computer vision project that automatically aligns scanned documents with their reference template using feature matching and homography transformation.

## Overview

This project implements an automated document alignment system that can align a scanned document with a reference template image. The algorithm uses ORB features, keypoint matching, and projective transformation to achieve precise alignment.

## Algorithm

The alignment process follows these steps:
1. **Read Images**: Load reference template and scanned document
2. **Feature Detection**: Extract ORB keypoints and descriptors (up to 500 features)
3. **Feature Matching**: Match corresponding keypoints using Hamming distance
4. **Homography Estimation**: Compute 3x3 transformation matrix using RANSAC
5. **Warp & Display**: Apply projective transformation to align the scanned image

## Features

- Automatic keypoint detection (ORB)
- Robust feature matching with top 10% best matches
- RANSAC-based homography estimation for outlier rejection
- Perspective transformation warping
- Visual comparison of aligned results

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib

## Usage

Open the Jupyter notebook `Image_Alignment.ipynb` and run all cells. Place your reference image as `images/form.jpg` and scanned image as `images/scanned_form.jpg`.
