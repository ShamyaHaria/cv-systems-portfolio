CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
BIN_DIR = ../bin
INC_DIR = ../include

# OpenCV configuration
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

# ONNX Runtime configuration
ONNX_INC = /usr/local/include/onnxruntime
ONNX_LIB = /usr/local/lib
ONNX_LIBS = -lonnxruntime

# Combined includes and libraries
INCLUDES = -I$(INC_DIR) $(OPENCV_CFLAGS) -I$(ONNX_INC)
LIBS = $(OPENCV_LIBS) -L$(ONNX_LIB) $(ONNX_LIBS)

# Source files
SOURCES = vidDisplay.cpp filters.cpp depthEstimator.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = $(BIN_DIR)/opencv_app

all: $(BIN_DIR) $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) -Wl,-rpath,/usr/local/lib
	@echo "Build complete!"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

rebuild: clean all

run: $(TARGET)
	cd $(BIN_DIR) && ./opencv_app

.PHONY: all clean rebuild run