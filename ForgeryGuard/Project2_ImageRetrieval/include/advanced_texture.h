/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Advanced texture feature extraction using co-occurrence matrices, Gabor filters, and Laws filters
 */

#ifndef ADVANCED_TEXTURE_H
#define ADVANCED_TEXTURE_H

#include <opencv2/opencv.hpp>
#include <vector>

// Stores four Haralick features extracted from co-occurrence matrix
struct CooccurrenceFeatures {
    float energy;
    float entropy;
    float contrast;
    float homogeneity;
};

// Computes co-occurrence matrix at specified distance and angle
cv::Mat computeCooccurrenceMatrix(const cv::Mat &image, int distance = 1, int angle = 0);

// Extracts energy, entropy, contrast, and homogeneity from co-occurrence matrix
CooccurrenceFeatures extractCooccurrenceFeatures(const cv::Mat &cooccurrence);

// Generates Gabor filter bank at multiple scales and orientations
std::vector<cv::Mat> generateGaborFilters(int num_scales = 4, int num_orientations = 6);

// Applies Gabor filters and computes histogram of filter responses
std::vector<float> extractGaborFeatures(const cv::Mat &image, int bins = 16);

// Generates Laws texture filters using L5, E5, S5, W5, R5 kernels
std::vector<cv::Mat> generateLawsFilters();

// Applies Laws filters and computes energy for each filter
std::vector<float> extractLawsFeatures(const cv::Mat &image);

// Combines all texture features into single 233-dimensional vector
std::vector<float> extractAdvancedTextureFeature(const cv::Mat &image);

#endif