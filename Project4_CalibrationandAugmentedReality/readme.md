# Project 4: Camera Calibration and Augmented Reality
**CS 5330 Pattern Recognition and Computer Vision**
Khoury College of Computer Science, Northeastern University

**Author:** Shamya Haria
**Date:** March 2026

---

## Overview

This project implements a full camera calibration and augmented reality pipeline in C++ using OpenCV. The system detects a printed checkerboard target, calibrates the camera from multiple views, estimates the board's 3D pose in real time using solvePnP, and projects virtual 3D objects into the scene so they appear physically anchored to the board. A separate program demonstrates Harris corner feature detection on live video. Three extensions add static image AR, a second virtual object, and a texture overlay that visually replaces the calibration target.

---

## File Structure
```
Project4_CalibrationandAugmentedReality/
├── src/
│   ├── calibration.cpp       Tasks 1, 2, 3
│   ├── ar.cpp                Tasks 4, 5, 6 + Extensions 1 and 2
│   ├── features.cpp          Task 7
│   └── overlay.cpp           Extension 3
├── data/
│   └── overlay.jpg           Texture image used for overlay extension
├── results/                  Saved calibration frames and AR static outputs
├── CMakeLists.txt
├── calibration.yml           Generated after running calibration
└── README.md
```

---

## Build Instructions
```bash
cd Project4_CalibrationandAugmentedReality
mkdir build
cd build
cmake ..
make
```

This builds four executables inside the build folder.

---

## How to Run

### Step 1 — Calibration (run this first)
```bash
./build/calibration
```
Loads 9 saved calibration images from the results folder, detects corners, calibrates the camera, and saves parameters to calibration.yml. Also opens a live camera window where you can press s to save new frames and c to recalibrate.

Controls:
- `s` — save current frame
- `c` — run calibration (needs at least 5 frames)
- `q` — quit

---

### Step 2 — Augmented Reality
```bash
./build/ar
```
Reads calibration.yml, opens live camera, detects the board each frame, runs solvePnP, and projects virtual objects onto the scene.

Controls:
- `a` — toggle 3D axes
- `o` — toggle house
- `r` — toggle rocket
- `c` — toggle outer corner rectangle
- `q` — quit

**Static image mode (Extension 1):**
```bash
./build/ar results/Tilt1.png results/Tilt2.png
```
Pass any number of image paths as arguments. AR is projected onto each saved image and the result is saved to results/.

---

### Step 3 — Harris Feature Detection
```bash
./build/features
```
Opens live camera and runs Harris corner detection on every frame at 640x480.

Controls:
- `+` — increase threshold
- `-` — decrease threshold
- `q` — quit

---

### Step 4 — Texture Overlay (Extension 3)
```bash
./build/overlay
```
Reads calibration.yml and opens live camera. Detects the board and warps the overlay image onto it each frame, replacing the checkerboard visually while AR objects continue to track above it.

Controls:
- `t` — toggle texture overlay
- `a` — toggle axes
- `o` — toggle house
- `q` — quit

---

## Tasks Completed

| Task | Description | File |
|------|-------------|------|
| Task 1 | Detect and extract chessboard corners | calibration.cpp |
| Task 2 | Select and save calibration frames | calibration.cpp |
| Task 3 | Calibrate camera, save to file | calibration.cpp |
| Task 4 | solvePnP pose estimation, print R and T | ar.cpp |
| Task 5 | Project 3D axes and outer corners | ar.cpp |
| Task 6 | Virtual house floating above board | ar.cpp |
| Task 7 | Harris corner feature detection | features.cpp |

---

## Extensions

| Extension | Description | File |
|-----------|-------------|------|
| Static Images | AR on pre-captured images via command line args | ar.cpp |
| Rocket Object | Second toggleable virtual object with fins and exhaust | ar.cpp |
| Texture Overlay | Replace checkerboard visually using homography warp | overlay.cpp |

---

## Calibration Results

Camera used: iPhone 15 Pro Max via macOS Continuity Camera
Frames used: 7 valid frames out of 9 collected
Reprojection error: 1.235 pixels

Camera matrix after calibration:
```
[2538.78,    0,    922.67]
[   0,    2538.78, 510.10]
[   0,       0,      1  ]
```

Distortion coefficients: [0.110, 0.466, -0.002, -0.005, -2.492]

---

## Dependencies

- OpenCV 4.x
- CMake 3.10 or higher
- C++17

Install OpenCV on macOS:
```bash
brew install opencv
```

---

## Video URLs

N/A

---

## Time Travel Days Used

0

---

## Acknowledgements

- Professor Bruce Maxwell for project specification and checkerboard image
- OpenCV documentation at docs.opencv.org
- Claude by Anthropic