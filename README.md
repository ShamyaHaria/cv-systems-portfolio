# CS5330 Pattern Recognition and Computer Vision
### Khoury College of Computer Sciences, Northeastern University
**Shamya Haria**

---

A collection of projects built for CS5330 Pattern Recognition and Computer Vision, covering real-time video processing, image retrieval, object recognition, augmented reality, deep learning, and handwriting forensics.

---

## Projects

### Project 1 — Real-Time Video Effects Engine
Built a real-time video processing pipeline in C++ using OpenCV. Implements a suite of image filters and effects applied live to webcam input including greyscale conversion, blur, Sobel edge detection, and custom artistic filters.

### Project 2 — Content-Based Image Retrieval
Designed an image retrieval system that matches a query image against a database using feature descriptors. Extended with adaptive feature weighting, saliency-based matching, and query refinement to improve retrieval accuracy beyond baseline histogram matching.

### Project 3 — Real-Time 2D Object Recognition
Built a real-time object recognition system using classical computer vision — custom thresholding, connected components, and Hu moment feature vectors. Extended with a CNN embedding pipeline using ResNet18 for comparison against the classical approach.

### Project 4 — Camera Calibration and Augmented Reality
Implemented full camera calibration using checkerboard detection and solvePnP. Used the calibrated camera to overlay virtual 3D objects (house, rocket) onto real scenes in real time, including texture mapping via homography.

### Project 5 — Recognition using Deep Networks
Explored deep network architectures for pattern recognition tasks. Covers network design, training, and evaluation on standard vision benchmarks.

### Final Project — ForgeryGuard: Handwriting Forgery Detection
Built a complete handwriting verification system using a Siamese Network trained with contrastive loss on the IAM Handwriting Database. Given two handwriting samples, the system determines whether they were written by the same person. Evaluated on both held-out IAM writers and a self-collected dataset of six unseen individuals.

- **IAM accuracy:** 55.16% | EER: 44.84%
- **Self-collected accuracy:** 71.43% | EER: 20.00%
- Includes Grad-CAM visualizations, ablation study, and edge case analysis

---

## Stack
C++, Python, PyTorch, OpenCV, torchvision, scikit-learn, Matplotlib

---

## Course
CS5330 Pattern Recognition and Computer Vision
Professor Bruce Maxwell — Spring 2026