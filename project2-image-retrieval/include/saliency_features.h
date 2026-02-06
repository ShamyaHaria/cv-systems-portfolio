/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Saliency-based feature extraction using spectral residual analysis
 */

#ifndef SALIENCY_FEATURES_H
#define SALIENCY_FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>

// Computes saliency map using spectral residual in Fourier domain
cv::Mat computeSaliencyMap(const cv::Mat& image);

// Computes saliency using contrast-based approach
cv::Mat computeGraphBasedSaliency(const cv::Mat& image);

// Extracts color histogram weighted by saliency values
std::vector<float> extractSaliencyWeightedHistogram(const cv::Mat& image,const cv::Mat& saliency_map,int bins_per_channel = 8);

// Extracts texture histogram weighted by saliency values
std::vector<float> extractSaliencyWeightedTexture(const cv::Mat& image,const cv::Mat& saliency_map,int bins = 16);

// Combines saliency-weighted color and texture features
std::vector<float> extractSaliencyFeature(const cv::Mat& image);

// Creates visualization overlaying saliency heat map on original image
cv::Mat visualizeSaliency(const cv::Mat& image, const cv::Mat& saliency_map);

#endif