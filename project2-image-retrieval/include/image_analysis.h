/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Image characteristic analysis for adaptive feature weighting
 */

#ifndef IMAGE_ANALYSIS_H
#define IMAGE_ANALYSIS_H

#include <opencv2/opencv.hpp>

// Stores computed characteristics of an image
struct ImageCharacteristics {
    float color_variance;
    float texture_strength;
    float spatial_complexity;
    float brightness_range;
};

// Analyzes image to compute color, texture, spatial, and brightness characteristics
ImageCharacteristics analyzeImage(const cv::Mat &image);

// Stores adaptive weights for different feature types
struct FeatureWeights {
    float color_weight;
    float texture_weight;
    float spatial_weight;
};

// Computes optimal feature weights based on image characteristics
FeatureWeights computeAdaptiveWeights(const ImageCharacteristics &chars);

// Computes color variance in HSV space
float computeColorVarianceHSV(const cv::Mat &image);

// Computes color distribution entropy using hue channel
float computeColorDistribution(const cv::Mat &image);

// Computes mean gradient magnitude as texture strength indicator
float computeTextureStrength(const cv::Mat &image);

// Computes ratio of edge pixels using Canny edge detection
float computeEdgeDensity(const cv::Mat &image);

// Computes variance across image grid regions
float computeSpatialComplexity(const cv::Mat &image);

// Computes grayscale histogram entropy
float computeEntropyMetric(const cv::Mat &image);

#endif