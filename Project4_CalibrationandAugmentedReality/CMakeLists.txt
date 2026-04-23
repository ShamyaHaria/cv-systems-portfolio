# Shamya Haria
# Project 4: Calibration and Augmented Reality

cmake_minimum_required(VERSION 3.10)
project(CalibrationAR)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(calibration src/calibration.cpp)
target_link_libraries(calibration ${OpenCV_LIBS})

add_executable(ar src/ar.cpp)
target_link_libraries(ar ${OpenCV_LIBS})

add_executable(features src/features.cpp)
target_link_libraries(features ${OpenCV_LIBS})

add_executable(overlay src/overlay.cpp)
target_link_libraries(overlay ${OpenCV_LIBS})