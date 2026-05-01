cmake_minimum_required(VERSION 3.10)
project(ImageRetrieval)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(cbir_common
    src/csv_utils.cpp
    src/distance_metrics.cpp
    src/feature_extraction.cpp
    src/advanced_texture.cpp
    src/image_analysis.cpp
    src/saliency_features.cpp
    src/query_refinement.cpp
)
target_link_libraries(cbir_common ${OpenCV_LIBS})

add_executable(baseline_matching src/baseline_matching.cpp)
target_link_libraries(baseline_matching cbir_common ${OpenCV_LIBS})

add_executable(histogram_matching src/histogram_matching.cpp)
target_link_libraries(histogram_matching cbir_common ${OpenCV_LIBS})

add_executable(multi_histogram src/multi_histogram.cpp)
target_link_libraries(multi_histogram cbir_common ${OpenCV_LIBS})

add_executable(texture_color src/texture_color.cpp)
target_link_libraries(texture_color cbir_common ${OpenCV_LIBS})

add_executable(dnn_matching src/dnn_matching.cpp)
target_link_libraries(dnn_matching cbir_common ${OpenCV_LIBS})

add_executable(custom_design src/custom_design.cpp)
target_link_libraries(custom_design cbir_common ${OpenCV_LIBS})

add_executable(advanced_texture_matching src/advanced_texture_matching.cpp)
target_link_libraries(advanced_texture_matching cbir_common ${OpenCV_LIBS})

add_executable(adaptive_matching src/adaptive_matching.cpp)
target_link_libraries(adaptive_matching cbir_common ${OpenCV_LIBS})

add_executable(saliency_matching src/saliency_matching.cpp)
target_link_libraries(saliency_matching cbir_common ${OpenCV_LIBS})

add_executable(query_refinement_matching src/query_refinement_matching.cpp)
target_link_libraries(query_refinement_matching cbir_common ${OpenCV_LIBS})

add_executable(yolo_detector src/yolo_object_detector.cpp)
target_link_libraries(yolo_detector cbir_common ${OpenCV_LIBS})

add_executable(shape_detector src/shape_based_detector.cpp)
target_link_libraries(shape_detector cbir_common ${OpenCV_LIBS})

add_executable(mobilenet_detector src/mobilenet_detector.cpp)
target_link_libraries(mobilenet_detector cbir_common ${OpenCV_LIBS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)