# Shamya Haria
## CS5330 - Project 3: Real-time 2D Object Recognition

---

## VIDEO DEMO LINK

https://drive.google.com/file/d/1xaeLseg31xKex30qBnEuUnJXv-ahporJ/view?usp=sharing

---

## SYSTEM INFORMATION

- **Operating System:** macOS (MacBook Air M1/M2)
- **IDE:** Visual Studio Code, Terminal, nano
- **OpenCV Version:** 4.13.0
- **C++ Standard:** C++17

---

## BUILD INSTRUCTIONS

1. Navigate to project directory
2. Create build directory and compile:
```bash
mkdir build
cd build
cmake ..
make
cd ..
```

---

## RUNNING THE SYSTEM

### Real-Time Video Recognition (Main System)
```bash
./build/realtime
```

**Keyboard Controls:**
- `t` : Toggle between training and recognition modes
- `n` : Save current object to database (training mode only)
- `s` : Save screenshots of all windows
- `q` : Quit program

**Setup Requirements:**
- Webcam or built-in laptop camera
- White paper or white wall background
- Dark colored objects for detection
- Good lighting conditions

### Batch Training (Extension 1)
```bash
./build/batch_train
```

Automatically trains all images in `data/train/` folder. Labels are extracted from filenames (e.g., `pen_1.jpg` → `pen`).

### CNN Embedding Evaluation (Extension 2)
```bash
python3 embedding_system.py
```

**Requires:**
- ResNet18 ONNX model (`resnet18.onnx`)
- Python libraries: `opencv-python`, `numpy`
- Training images in `data/train/`
- Test images in `data/test/` (optional)

### Embedding Visualization (Extension 3)
```bash
python3 visualize_embeddings.py
```

Generates 2D PCA plot saved to `results/embedding_visualization.png`

**Requires:** `matplotlib`, `scikit-learn`

---

## EXTENSIONS IMPLEMENTED

### 1. Batch Training System (`batch_train.cpp`)
- Automated labeling from filenames
- Processes entire dataset in <10 seconds
- Eliminates manual labeling errors

### 2. Recognition of 10 Object Categories
**Original 5:** triangle, hammer, allen_key, screwdriver, key_fob  
**Added 5:** cd, phone, star, postit, scissors  
Maintains 80% accuracy with expanded object set

### 3. 2D Embedding Visualization (`visualize_embeddings.py`)
- PCA projection of 512-dim ResNet18 features
- Demonstrates object clustering in feature space
- 28.28% variance explained by first 2 components

---

## DEPENDENCIES

### C++ Libraries
- OpenCV 4.13.0+ (with DNN module)
- C++17 standard library
- Filesystem library

### Python Libraries
- opencv-python (cv2)
- numpy
- matplotlib
- scikit-learn

### Pre-trained Model
- ResNet18 ONNX: `resnet18.onnx`  
  Download from ONNX Model Zoo if not included

---

## FILE STRUCTURE
```
Project3_RealTime2DObjectDetection/
├── main.cpp                    # Real-time video recognition
├── processing.cpp/.h           # Core processing functions
├── batch_train.cpp             # Automated batch training
├── embedding_system.py         # CNN evaluation script
├── visualize_embeddings.py     # PCA visualization
├── CMakeLists.txt              # Build configuration
├── object_db.csv              # Feature database (generated)
├── resnet18.onnx              # Pre-trained model
├── results/                    # Output visualizations
├── screenshots/                # Saved screenshots
└── README.md                   # This file
```

---

## NOTES

- The system requires good lighting and high contrast between objects and background for optimal detection
- Threshold value (default 100) may need adjustment based on lighting conditions - modify in `main.cpp` line 37
- Minimum object area filter (default 1000 pixels) can be adjusted in `main.cpp` line 112 for smaller objects
- CNN evaluation requires `resnet18.onnx` model file

---

## TIME TRAVEL DAYS

**Time Travel Days Used:** 0

---