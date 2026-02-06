/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Advanced texture analysis using GLCM, Gabor filters, and Laws filters
 */

#include "advanced_texture.h"
#include <cmath>
#include <algorithm>

cv::Mat computeCooccurrenceMatrix(const cv::Mat& image, int distance, int angle) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Quantize to 16 levels to reduce matrix size
    int levels = 16;
    cv::Mat quantized;
    gray.convertTo(quantized, CV_32S);
    quantized = quantized / (256 / levels);
    quantized = cv::max(quantized, 0);
    quantized = cv::min(quantized, levels - 1);
    
    cv::Mat cooccurrence = cv::Mat::zeros(levels, levels, CV_32F);
    
    // Compute offset based on angle
    int dx = 0, dy = 0;
    if (angle == 0) { dx = distance; dy = 0; }
    else if (angle == 45) { dx = distance; dy = -distance; }
    else if (angle == 90) { dx = 0; dy = -distance; }
    else if (angle == 135) { dx = -distance; dy = -distance; }
    
    // Build co-occurrence matrix
    for (int i = 0; i < quantized.rows; i++) {
        for (int j = 0; j < quantized.cols; j++) {
            int ni = i + dy;
            int nj = j + dx;
            
            if (ni >= 0 && ni < quantized.rows && nj >= 0 && nj < quantized.cols) {
                int val1 = quantized.at<int>(i, j);
                int val2 = quantized.at<int>(ni, nj);
                
                if (val1 >= 0 && val1 < levels && val2 >= 0 && val2 < levels) {
                    cooccurrence.at<float>(val1, val2)++;
                }
            }
        }
    }
    
    // Normalize to probabilities
    float sum = cv::sum(cooccurrence)[0];
    if (sum > 0) {
        cooccurrence /= sum;
    }
    
    return cooccurrence;
}

CooccurrenceFeatures extractCooccurrenceFeatures(const cv::Mat& cooccurrence) {
    CooccurrenceFeatures features;
    features.energy = 0;
    features.entropy = 0;
    features.contrast = 0;
    features.homogeneity = 0;
    
    // Compute Haralick features
    for (int i = 0; i < cooccurrence.rows; i++) {
        for (int j = 0; j < cooccurrence.cols; j++) {
            float val = cooccurrence.at<float>(i, j);
            
            if (val > 0) {
                features.energy += val * val;
                features.entropy -= val * std::log(val + 1e-10);
                features.contrast += (i - j) * (i - j) * val;
                features.homogeneity += val / (1 + (i - j) * (i - j));
            }
        }
    }
    
    return features;
}

std::vector<cv::Mat> generateGaborFilters(int num_scales, int num_orientations) {
    std::vector<cv::Mat> filters;
    int ksize = 31;
    double sigma = 4.0;
    
    // Generate filters at multiple scales and orientations
    for (int s = 0; s < num_scales; s++) {
        double lambda = 8.0 * std::pow(2.0, s);
        
        for (int o = 0; o < num_orientations; o++) {
            double theta = o * CV_PI / num_orientations;
            
            cv::Mat gabor = cv::getGaborKernel(
                cv::Size(ksize, ksize), sigma, theta, lambda, 0.5, 0, CV_32F
            );
            
            filters.push_back(gabor);
        }
    }
    
    return filters;
}

std::vector<float> extractGaborFeatures(const cv::Mat& image, int bins) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    gray.convertTo(gray, CV_32F);
    auto filters = generateGaborFilters(4, 6);
    std::vector<float> features;
    
    // Apply each Gabor filter and compute response histogram
    for (const auto& filter : filters) {
        cv::Mat filtered;
        cv::filter2D(gray, filtered, CV_32F, filter);
        
        cv::Mat magnitude = cv::abs(filtered);
        
        double minVal, maxVal;
        cv::minMaxLoc(magnitude, &minVal, &maxVal);
        
        std::vector<float> hist(bins, 0);
        float binWidth = (maxVal - minVal) / bins;
        
        if (binWidth > 0) {
            for (int i = 0; i < magnitude.rows; i++) {
                for (int j = 0; j < magnitude.cols; j++) {
                    float val = magnitude.at<float>(i, j);
                    int binIdx = std::min((int)((val - minVal) / binWidth), bins - 1);
                    hist[binIdx]++;
                }
            }
            
            float sum = 0;
            for (float h : hist) sum += h;
            if (sum > 0) {
                for (float& h : hist) h /= sum;
            }
        }
        
        features.insert(features.end(), hist.begin(), hist.end());
    }
    
    return features;
}

std::vector<cv::Mat> generateLawsFilters() {
    std::vector<cv::Mat> filters;
    
    // Define Laws 1D kernels
    std::vector<float> L5 = {1, 4, 6, 4, 1};
    std::vector<float> E5 = {-1, -2, 0, 2, 1};
    std::vector<float> S5 = {-1, 0, 2, 0, -1};
    std::vector<float> W5 = {-1, 2, 0, -2, 1};
    std::vector<float> R5 = {1, -4, 6, -4, 1};
    
    std::vector<std::vector<float>> kernels = {L5, E5, S5, W5, R5};
    
    // Create 2D filters via outer products
    for (size_t i = 0; i < kernels.size(); i++) {
        for (size_t j = 0; j < kernels.size(); j++) {
            cv::Mat filter(5, 5, CV_32F);
            for (int r = 0; r < 5; r++) {
                for (int c = 0; c < 5; c++) {
                    filter.at<float>(r, c) = kernels[i][r] * kernels[j][c];
                }
            }
            filters.push_back(filter);
        }
    }
    
    return filters;
}

std::vector<float> extractLawsFeatures(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    gray.convertTo(gray, CV_32F);
    auto filters = generateLawsFilters();
    std::vector<float> features;
    
    // Apply each Laws filter and compute energy
    for (const auto& filter : filters) {
        cv::Mat filtered;
        cv::filter2D(gray, filtered, CV_32F, filter);
        
        // Energy is sum of squared filter responses
        float energy = 0;
        for (int i = 0; i < filtered.rows; i++) {
            for (int j = 0; j < filtered.cols; j++) {
                float val = filtered.at<float>(i, j);
                energy += val * val;
            }
        }
        
        features.push_back(energy);
    }
    
    return features;
}

std::vector<float> extractAdvancedTextureFeature(const cv::Mat& image) {
    std::vector<float> features;
    
    // Extract co-occurrence features at 4 orientations
    for (int angle = 0; angle < 180; angle += 45) {
        cv::Mat cooccurrence = computeCooccurrenceMatrix(image, 1, angle);
        CooccurrenceFeatures cooc_feat = extractCooccurrenceFeatures(cooccurrence);
        
        features.push_back(cooc_feat.energy);
        features.push_back(cooc_feat.entropy);
        features.push_back(cooc_feat.contrast);
        features.push_back(cooc_feat.homogeneity);
    }
    
    // Add Gabor filter features
    auto gabor_feat = extractGaborFeatures(image, 8);
    features.insert(features.end(), gabor_feat.begin(), gabor_feat.end());
    
    // Add Laws filter features
    auto laws_feat = extractLawsFeatures(image);
    features.insert(features.end(), laws_feat.begin(), laws_feat.end());
    
    return features;
}