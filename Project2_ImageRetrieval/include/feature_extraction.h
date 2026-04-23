/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Feature extraction functions for various CBIR matching methods
 */

#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Extracts 7x7 center square as 147-element feature vector (Task 1)
std::vector<float> extractBaselineFeature(const cv::Mat &image);

// Extracts rg chromaticity histogram with specified bins (Task 2)
std::vector<float> extractRGChromaticityHistogram(const cv::Mat &image, int r_bins = 16, int g_bins = 16);

// Extracts RGB color histogram with specified bins per channel (Task 2)
std::vector<float> extractRGBHistogram(const cv::Mat &image, int bins_per_channel = 8);

// Extracts separate RGB histograms for top and bottom image halves (Task 3)
std::vector<float> extractMultiRegionHistogram(const cv::Mat &image, int bins_per_channel = 8);

// Extracts histogram of Sobel gradient magnitudes (Task 4)
std::vector<float> extractGradientMagnitudeHistogram(const cv::Mat &image, int bins = 16);

// Combines RGB color histogram with gradient magnitude histogram (Task 4)
std::vector<float> extractColorTextureFeature(const cv::Mat &image, int color_bins = 8, int texture_bins = 16);

// Applies Sobel X and Y filters and computes gradient magnitude
cv::Mat computeSobelMagnitude(const cv::Mat &image);

// Computes histogram from single-channel image with specified range
std::vector<float> computeHistogram(const cv::Mat &image, int bins, float min_val = 0.0f, float max_val = 255.0f);

#endif