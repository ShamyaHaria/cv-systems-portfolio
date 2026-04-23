# CS5330 – Pattern Recognition and Computer Vision
# Project 5: Recognition using Deep Networks
**Shamya Haria**
Northeastern University
Spring 2026

---

## Project Overview

This project explores building, training, and analyzing deep neural networks
for image recognition. I start with a CNN for MNIST digit classification, then
extend it with filter visualization, transfer learning for Greek letters,
a Vision Transformer implementation, and a hyperparameter experiment.

---

## Files

### Core Tasks
- task1_train.py — Builds, trains, and saves the MNIST CNN
- task1_evaluate.py — Evaluates on the first 10 test examples
- task1_custom_digits.py — Tests on my own handwritten digit photos
- task2_examine.py — Visualizes conv1 filters and their effects
- task3_greek.py — Transfer learning on Greek letters (alpha, beta, gamma)
- task4_transformer.py — Vision Transformer for MNIST
- task5_experiment.py — Hyperparameter experiment (3 dimensions)

### Extensions
- extension1_gabor.py — Fixed Gabor filter bank as first conv layer
- extension2_resnet_analysis.py — Pre-trained ResNet18 filter analysis
- extension3_live_recognition.py — Live webcam digit recognition

### Other
- run_all.py — Runs all tasks in sequence

---

## How to Run

### Requirements
Install dependencies:
pip install torch torchvision opencv-python-headless matplotlib pillow

### Running individual tasks
python3.11 task1_train.py
python3.11 task1_evaluate.py
python3.11 task1_custom_digits.py
python3.11 task2_examine.py
python3.11 task3_greek.py
python3.11 task4_transformer.py
python3.11 task5_experiment.py
python3.11 extension1_gabor.py
python3.11 extension2_resnet_analysis.py
python3.11 extension3_live_recognition.py

### Running all at once
python3.11 run_all.py

### Notes
- Run task1_train.py first since everything else uses the saved model
- Task 3 needs the greek_train/ folder with alpha/, beta/, gamma/ subdirectories
- Task 1c needs digit photos (digit_0.png through digit_9.png) in outputs/custom_digits/
- Extension 3 needs a webcam
- All output goes to outputs/

---

## Greek Letter Test Images

My handwritten alpha, beta, and gamma photos for Task 3 testing:
[https://drive.google.com/file/d/1FY1VKKXgQ76lWMiSU4oNkJZvlx0C56qA/view?usp=sharing]

---

## Time Travel Days

Not using any time travel days.

---

## Environment

- MacBook Air, macOS
- Python 3.11
- PyTorch + torchvision
- OpenCV for image filtering and contour detection

---

## Acknowledgements

- MNIST dataset (LeCun, Cortes, Burges)
- Greek letter dataset from CS 5330 course materials
- PyTorch documentation and tutorials
- OpenCV documentation
- Professor Bruce Maxwell for course materials and guidance