/*
 * Shamya Haria
 * January 26, 2026
 *
 * CS 5330 - Project 1
 * 
 * imgDisplay.cpp
 * Loads and displays a single image file with basic keyboard controls.
 * Warmup program to familiarize with OpenCV image loading and display functions.
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string imagePath = "../data/test_image.jpg";
    
    // Accept image path from command line if provided
    if (argc > 1) {
        imagePath = argv[1];
    }
    
    cv::Mat image = cv::imread(imagePath);
    
    // Verify image loaded successfully
    if (image.empty()) {
        std::cout << "ERROR: Could not load image: " << imagePath << std::endl;
        std::cout << "Usage: " << argv[0] << " [image_path]" << std::endl;
        return -1;
    }

    std::cout << "Loaded: " << imagePath << std::endl;
    std::cout << "Size: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "\nControls:" << std::endl;
    std::cout << "  q - Quit" << std::endl;
    std::cout << "  i - Show image info" << std::endl;
    std::cout << "  s - Save copy" << std::endl;
    
    cv::namedWindow("Image Display", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image Display", image);
    
    // Event loop for keyboard input
    while (true) {
        char key = cv::waitKey(0);
        
        if (key == 'q') {
            std::cout << "Quitting..." << std::endl;
            break;
        }
        else if (key == 'i') {
            std::cout << "\n--- Image Information ---" << std::endl;
            std::cout << "Dimensions: " << image.cols << "x" << image.rows << std::endl;
            std::cout << "Channels: " << image.channels() << std::endl;
            std::cout << "Total pixels: " << image.total() << std::endl;
        }
        else if (key == 's') {
            std::string savePath = "../data/saved_image.jpg";
            cv::imwrite(savePath, image);
            std::cout << "Saved to: " << savePath << std::endl;
        }
    }
    
    cv::destroyAllWindows();
    return 0;
}