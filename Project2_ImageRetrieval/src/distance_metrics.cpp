/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Distance metric implementations for feature vector comparison
 */

#include "distance_metrics.h"
#include <cmath>
#include <numeric>
#include <iostream>

float sumSquaredDifference(const std::vector<float>& f1, const std::vector<float>& f2) {
    if (f1.size() != f2.size()) {
        std::cerr << "Error: Feature vectors have different sizes" << std::endl;
        return -1.0f;
    }
    
    // Accumulate squared differences
    float ssd = 0.0f;
    for (size_t i = 0; i < f1.size(); i++) {
        float diff = f1[i] - f2[i];
        ssd += diff * diff;
    }
    
    return ssd;
}

float histogramIntersection(const std::vector<float>& h1, const std::vector<float>& h2) {
    if (h1.size() != h2.size()) {
        std::cerr << "Error: Histograms have different sizes" << std::endl;
        return 0.0f;
    }
    
    // Sum minimum values at each bin
    float intersection = 0.0f;
    for (size_t i = 0; i < h1.size(); i++) {
        intersection += std::min(h1[i], h2[i]);
    }
    
    return intersection;
}

float histogramIntersectionDistance(const std::vector<float>& h1, const std::vector<float>& h2) {
    // Normalize histograms to sum to 1
    auto h1_norm = normalizeHistogram(h1);
    auto h2_norm = normalizeHistogram(h2);
    
    float intersection = histogramIntersection(h1_norm, h2_norm);
    
    // Convert intersection to distance metric
    return 1.0f - intersection;
}

float cosineDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
    if (v1.size() != v2.size()) {
        std::cerr << "Error: Vectors have different sizes" << std::endl;
        return 2.0f;
    }
    
    // Normalize both vectors to unit length
    auto v1_norm = normalizeVector(v1);
    auto v2_norm = normalizeVector(v2);
    
    // Compute dot product (cosine of angle)
    float dot_product = 0.0f;
    for (size_t i = 0; i < v1_norm.size(); i++) {
        dot_product += v1_norm[i] * v2_norm[i];
    }
    
    // Clamp to handle floating point precision errors
    dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
    
    return 1.0f - dot_product;
}

float euclideanDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
    if (v1.size() != v2.size()) {
        std::cerr << "Error: Vectors have different sizes" << std::endl;
        return -1.0f;
    }
    
    // Compute sum of squared differences and take square root
    float sum = 0.0f;
    for (size_t i = 0; i < v1.size(); i++) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    
    return std::sqrt(sum);
}

std::vector<float> normalizeVector(const std::vector<float>& vec) {
    std::vector<float> normalized(vec.size());
    
    // Calculate L2 norm
    float norm = 0.0f;
    for (float val : vec) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    
    if (norm < 1e-10f) {
        return vec;
    }
    
    // Divide each element by norm
    for (size_t i = 0; i < vec.size(); i++) {
        normalized[i] = vec[i] / norm;
    }
    
    return normalized;
}

std::vector<float> normalizeHistogram(const std::vector<float>& hist) {
    std::vector<float> normalized(hist.size());
    
    float sum = std::accumulate(hist.begin(), hist.end(), 0.0f);
    
    if (sum < 1e-10f) {
        return hist;
    }
    
    // Normalize so histogram sums to 1
    for (size_t i = 0; i < hist.size(); i++) {
        normalized[i] = hist[i] / sum;
    }
    
    return normalized;
}