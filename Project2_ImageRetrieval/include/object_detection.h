/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Color-based object detection functions
 */

#ifndef OBJECT_DETECTION_H
#define OBJECT_DETECTION_H

#include <opencv2/opencv.hpp>
#include <vector>

// Computes banana detection score using yellow color and shape analysis
float detectBananaScore(const cv::Mat &image);

// Computes blue bin detection score using blue color dominance
float detectBlueBinScore(const cv::Mat &image);

// Extracts ratio of yellow pixels in HSV color space
float extractYellowRatio(const cv::Mat &image);

// Extracts ratio of blue pixels in HSV color space
float extractBlueRatio(const cv::Mat &image);

// Computes color variance across image regions
float computeColorVariance(const cv::Mat &image);

#endif