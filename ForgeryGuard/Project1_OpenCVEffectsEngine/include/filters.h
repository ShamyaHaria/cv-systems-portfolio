/*
 * Shamya Haria
 * January 26, 2026
 *
 * CS 5330 - Project 1
 * 
 * filters.h
 * Header file declaring all image filter and effect functions.
 * Includes basic filters (grayscale, sepia, blur), edge detection (Sobel),
 * face detection integration, depth-based effects, and creative filters.
 */

#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

// Custom grayscale conversion using inverted red channel
int greyscale(cv::Mat &src, cv::Mat &dst);

// Sepia tone filter for vintage photograph effect
int sepia(cv::Mat &src, cv::Mat &dst);

// Naive 5x5 Gaussian blur using full kernel
int blur5x5_1(cv::Mat &src, cv::Mat &dst);

// Optimized 5x5 Gaussian blur using separable filters
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

// Sobel X filter for vertical edge detection (returns signed short)
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

// Sobel Y filter for horizontal edge detection (returns signed short)
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

// Compute gradient magnitude from Sobel X and Y outputs
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

// Cartoon effect combining blur, color quantization, and edge darkening
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

// Detect faces in frame using Haar cascade classifier
int detectFaces(cv::Mat &frame, std::vector<cv::Rect> &faces);

// Portrait mode effect using depth map to selectively blur background
int depthFocusEffect(cv::Mat &src, cv::Mat &depth, cv::Mat &dst);

// Sketch filter creating pencil drawing effect from edges
int sketchFilter(cv::Mat &src, cv::Mat &dst);

// Spotlight effect darkening surroundings while keeping faces bright
int spotlightFace(cv::Mat &src, std::vector<cv::Rect> &faces, cv::Mat &dst);

// Glitch effect simulating analog TV interference with noise and scanlines
int glitchEffect(cv::Mat &src, cv::Mat &dst);

// Color pop effect isolating one color channel (0=blue, 1=green, 2=red)
int colorPop(cv::Mat &src, cv::Mat &dst, int channelToKeep);

// Spider-Man mask overlay aligned with detected faces
int spidermanMask(cv::Mat &src, std::vector<cv::Rect> &faces, cv::Mat &dst);

// Performance testing function comparing blur implementations
void testBlurTiming(cv::Mat &testImage);

#endif