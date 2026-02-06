/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Image characteristic analysis for adaptive feature weight computation
 */

#include "image_analysis.h"
#include <cmath>
#include <algorithm>

float computeColorVarianceHSV(const cv::Mat &image) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);

    // Compute standard deviation of hue and saturation
    cv::Scalar mean_h, stddev_h, mean_s, stddev_s;
    cv::meanStdDev(channels[0], mean_h, stddev_h);
    cv::meanStdDev(channels[1], mean_s, stddev_s);

    // Combine and normalize variance metrics
    float variance = (stddev_h[0] / 180.0) * 0.5 + (stddev_s[0] / 255.0) * 0.5;

    return variance;
}

float computeColorDistribution(const cv::Mat &image) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);

    // Build hue histogram
    int histSize = 32;
    float range[] = {0, 180};
    const float *histRange = {range};
    cv::Mat hist;
    cv::calcHist(&channels[0], 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    hist /= (image.rows * image.cols);

    // Calculate entropy
    float entropy = 0;
    for (int i = 0; i < histSize; i++) {
        float p = hist.at<float>(i);
        if (p > 0) {
            entropy -= p * std::log2(p);
        }
    }

    return entropy / std::log2(histSize);
}

float computeTextureStrength(const cv::Mat &image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // Compute gradients using Sobel
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);

    cv::Mat magnitude;
    cv::magnitude(grad_x, grad_y, magnitude);

    // Return normalized mean gradient
    cv::Scalar mean_mag = cv::mean(magnitude);
    return mean_mag[0] / 255.0;
}

float computeEdgeDensity(const cv::Mat &image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);

    // Calculate ratio of edge pixels
    int edge_pixels = cv::countNonZero(edges);
    int total_pixels = edges.rows * edges.cols;

    return (float)edge_pixels / total_pixels;
}

float computeSpatialComplexity(const cv::Mat &image) {
    int grid_size = 4;
    int region_height = image.rows / grid_size;
    int region_width = image.cols / grid_size;

    std::vector<float> region_means;

    // Compute mean brightness for each grid cell
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            cv::Rect region(j * region_width, i * region_height, region_width, region_height);
            cv::Mat roi = image(region);

            cv::Scalar mean = cv::mean(roi);
            float gray_mean = (mean[0] + mean[1] + mean[2]) / 3.0;
            region_means.push_back(gray_mean);
        }
    }

    // Calculate variance across regions
    float sum = 0, sum_sq = 0;
    for (float val : region_means) {
        sum += val;
        sum_sq += val * val;
    }
    float mean = sum / region_means.size();
    float variance = (sum_sq / region_means.size()) - (mean * mean);

    return std::sqrt(variance) / 255.0;
}

float computeEntropyMetric(const cv::Mat &image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // Compute grayscale histogram
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    hist /= (gray.rows * gray.cols);

    // Calculate Shannon entropy
    float entropy = 0;
    for (int i = 0; i < histSize; i++) {
        float p = hist.at<float>(i);
        if (p > 0) {
            entropy -= p * std::log2(p);
        }
    }

    return entropy / 8.0;
}

ImageCharacteristics analyzeImage(const cv::Mat &image) {
    ImageCharacteristics chars;

    // Analyze color properties
    chars.color_variance = computeColorVarianceHSV(image);
    float color_dist = computeColorDistribution(image);

    // Analyze texture properties
    chars.texture_strength = computeTextureStrength(image);
    float edge_density = computeEdgeDensity(image);

    // Analyze spatial properties
    chars.spatial_complexity = computeSpatialComplexity(image);

    // Analyze brightness range
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    double minVal, maxVal;
    cv::minMaxLoc(gray, &minVal, &maxVal);
    chars.brightness_range = (maxVal - minVal) / 255.0;

    // Average metrics
    chars.color_variance = (chars.color_variance + color_dist) / 2.0;
    chars.texture_strength = (chars.texture_strength + edge_density) / 2.0;

    return chars;
}

FeatureWeights computeAdaptiveWeights(const ImageCharacteristics &chars) {
    FeatureWeights weights;

    // Calculate importance scores for each feature type
    float color_importance = chars.color_variance * 0.6 + chars.brightness_range * 0.4;
    float texture_importance = chars.texture_strength;
    float spatial_importance = chars.spatial_complexity;

    float total = color_importance + texture_importance + spatial_importance;

    if (total > 0) {
        weights.color_weight = color_importance / total;
        weights.texture_weight = texture_importance / total;
        weights.spatial_weight = spatial_importance / total;
    } else {
        weights.color_weight = 0.33;
        weights.texture_weight = 0.33;
        weights.spatial_weight = 0.34;
    }

    // Enforce minimum 10% weight per feature
    weights.color_weight = std::max(0.1f, weights.color_weight);
    weights.texture_weight = std::max(0.1f, weights.texture_weight);
    weights.spatial_weight = std::max(0.1f, weights.spatial_weight);

    // Renormalize to sum to 1
    total = weights.color_weight + weights.texture_weight + weights.spatial_weight;
    weights.color_weight /= total;
    weights.texture_weight /= total;
    weights.spatial_weight /= total;

    return weights;
}