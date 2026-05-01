# Shamya Haria
# February 23, 2026
# Build configuration for real-time object recognition

cmake_minimum_required(VERSION 3.10)
project(RealtimeObjectRecognition)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(realtime main.cpp processing.cpp)
target_link_libraries(realtime ${OpenCV_LIBS})

add_executable(batch_train batch_train.cpp processing.cpp)
target_link_libraries(batch_train ${OpenCV_LIBS})