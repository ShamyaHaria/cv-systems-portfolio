PROJECT 2: CONTENT-BASED IMAGE RETRIEVAL SYSTEM
CS 5330 - Pattern Recognition & Computer Vision
Northeastern University

Author: Shamya Haria
Date: February 2025

--------------------------------------------------------------------------------

PROJECT DESCRIPTION

This project implements a comprehensive content-based image retrieval (CBIR) 
system that searches a database of 1,106 images to find visually similar matches 
to a query image. The system includes 7 required matching methods and 6 advanced 
extensions demonstrating both classical computer vision techniques and modern 
deep learning approaches.

--------------------------------------------------------------------------------

OPERATING SYSTEM & DEVELOPMENT ENVIRONMENT

OS: macOS (Apple Silicon M1/M2)
IDE: Visual Studio Code / Terminal
Compiler: clang++ (Apple clang version 15.x)
Build System: CMake 3.10+
Libraries: OpenCV 4.13.0

--------------------------------------------------------------------------------

COMPILATION INSTRUCTIONS

1. Navigate to project directory:
   cd project2-image-retrieval

2. Create build directory:
   mkdir build
   cd build

3. Configure with CMake:
   cmake ..

4. Compile all executables:
   make

This will create 13 executable programs in the build/ directory.

--------------------------------------------------------------------------------

RUNNING THE PROGRAMS

All executables follow similar command-line syntax:

REQUIRED TASKS:

Task 1 - Baseline Matching (7x7 SSD):
./baseline_matching <target_image> <database_directory> <num_results>
Example: ./baseline_matching ../data/olympus/pic.1016.jpg ../data/olympus 5

Task 2 - Histogram Matching (rg Chromaticity):
./histogram_matching <target_image> <database_directory> <num_results>
Example: ./histogram_matching ../data/olympus/pic.0164.jpg ../data/olympus 5

Task 3 - Multi-Histogram Matching (Spatial):
./multi_histogram <target_image> <database_directory> <num_results>
Example: ./multi_histogram ../data/olympus/pic.0274.jpg ../data/olympus 5

Task 4 - Texture + Color:
./texture_color <target_image> <database_directory> <num_results>
Example: ./texture_color ../data/olympus/pic.0535.jpg ../data/olympus 5

Task 5 - DNN Embeddings (ResNet18):
./dnn_matching <target_image> <embeddings_csv> <num_results>
Example: ./dnn_matching ../data/olympus/pic.0893.jpg ../data/embeddings.csv 5

Task 7 - Custom Design (Spatial + Texture + DNN):
./custom_design <target_image> <database_directory> <num_results> [embeddings_csv]
Example: ./custom_design ../data/olympus/pic.1072.jpg ../data/olympus 5 ../data/embeddings.csv

EXTENSIONS:

Extension 1 - Advanced Texture Features (GLCM + Gabor + Laws):
./advanced_texture_matching <target_image> <database_directory> <num_results>
Example: ./advanced_texture_matching ../data/olympus/pic.0535.jpg ../data/olympus 5

Extension 2 - Object Detection (MobileNet-SSD):
./mobilenet_detector <database_directory> <num_results> <object_class>
Available classes: person, chair, bottle, car, bicycle, bird, cat, dog, etc.
Example: ./mobilenet_detector ../data/olympus 10 person

Extension 3 - Adaptive Feature Weighting:
./adaptive_matching <target_image> <database_directory> <num_results>
Example: ./adaptive_matching ../data/olympus/pic.0164.jpg ../data/olympus 5

Extension 4 - Saliency-Based Matching:
./saliency_matching <target_image> <database_directory> <num_results> [save_visualization]
Example: ./saliency_matching ../data/olympus/pic.0164.jpg ../data/olympus 5 true

Extension 5 - Query Refinement (Interactive Learning):
./query_refinement_matching <target_image> <database_directory> <num_results> [iterations]
Example: ./query_refinement_matching ../data/olympus/pic.0164.jpg ../data/olympus 5 3

Extension 6 - Web GUI:
cd ../extensions/gui
python3 web_gui.py
Then open browser to: http://localhost:5000

--------------------------------------------------------------------------------

TESTING INSTRUCTIONS

To verify the implementation works correctly, test with the following commands:

Test Task 1 (should match project requirements exactly):
./baseline_matching ../data/olympus/pic.1016.jpg ../data/olympus 5
Expected top 3: pic.0986.jpg, pic.0641.jpg, pic.0547.jpg

Test Task 2:
./histogram_matching ../data/olympus/pic.0164.jpg ../data/olympus 5
Expected top 3: pic.0080.jpg, pic.1032.jpg, pic.0461.jpg

Test Task 3:
./multi_histogram ../data/olympus/pic.0274.jpg ../data/olympus 5
Expected top 3: pic.0273.jpg, pic.1031.jpg, pic.0409.jpg

Test Extension - Adaptive Weighting:
./adaptive_matching ../data/olympus/pic.0164.jpg ../data/olympus 5
Should display computed weights (e.g., Color: 77%, Texture: 12%, Spatial: 11%)

Test Extension - Object Detection:
./mobilenet_detector ../data/olympus 10 person
Should find ~10 images containing people with 99%+ confidence

Test Extension - Web GUI:
cd ../extensions/gui
python3 web_gui.py
Access at http://localhost:5000 and try different methods

--------------------------------------------------------------------------------

EXTENSION IMPLEMENTATIONS

All 6 extensions have been fully implemented:

1. Advanced Texture Features
   - Co-occurrence matrices (4 orientations, 4 Haralick features)
   - Gabor filter bank (24 filters: 4 scales x 6 orientations)
   - Laws texture filters (25 filter combinations)
   - Total: 233-dimensional feature vector

2. Object-Specific Detection (MobileNet-SSD)
   - Deep learning object detector
   - Detects 20 PASCAL VOC categories
   - Person, chair, and bottle detection demonstrated
   - 99%+ detection confidence

3. Adaptive Feature Weighting
   - Analyzes image characteristics (color variance, texture strength)
   - Automatically computes optimal feature weights
   - Outperforms fixed-weight combinations

4. Saliency-Based Matching
   - Spectral residual saliency computation
   - Weights features by visual attention
   - Includes heat map visualization

5. Query Refinement
   - Interactive relevance feedback
   - Iterative feature blending (alpha annealing)
   - Progressive learning from user selections

6. Web-Based GUI
   - Flask backend with C++ subprocess integration
   - Responsive grid layout
   - Supports all 7 matching methods

--------------------------------------------------------------------------------

PROJECT STRUCTURE

project2-image-retrieval/
├── CMakeLists.txt          (Build configuration)
├── README.md               (This file)
├── build/                  (Compiled executables)
├── data/
│   ├── olympus/           (1,106 image database)
│   ├── test_subset/       (15 test images)
│   └── embeddings.csv     (ResNet18 features)
├── docs/
│   └── report.pdf         (Project report)
├── extensions/
│   └── gui/               (Web interface)
├── include/               (8 header files)
├── models/                (MobileNet-SSD models)
├── results/               (Output visualizations)
├── scripts/               (Visualization utilities)
└── src/                   (20 C++ source files)

--------------------------------------------------------------------------------

DEPENDENCIES

Required:
- CMake 3.10 or higher
- C++17 compatible compiler
- OpenCV 4.x (with DNN module)

Optional (for Web GUI):
- Python 3.x
- Flask (pip3 install flask)

Installation on macOS:
brew install cmake opencv python3
pip3 install --break-system-packages flask

--------------------------------------------------------------------------------