/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Saliency map computation and saliency-weighted feature extraction
 */

#include "saliency_features.h"
#include <cmath>

cv::Mat computeSaliencyMap(const cv::Mat& image) {
    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    lab.convertTo(lab, CV_32F);
    
    cv::Mat saliency_map = cv::Mat::zeros(image.size(), CV_32F);
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    
    // Process each Lab channel in Fourier domain
    for (int c = 0; c < 3; c++) {
        cv::Mat padded;
        int m = cv::getOptimalDFTSize(channels[c].rows);
        int n = cv::getOptimalDFTSize(channels[c].cols);
        cv::copyMakeBorder(channels[c], padded, 0, m - channels[c].rows, 
                          0, n - channels[c].cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        
        // Forward FFT
        cv::Mat planes[] = {padded, cv::Mat::zeros(padded.size(), CV_32F)};
        cv::Mat complex_img;
        cv::merge(planes, 2, complex_img);
        cv::dft(complex_img, complex_img);
        
        // Compute magnitude and phase
        cv::split(complex_img, planes);
        cv::Mat magnitude, phase;
        cv::cartToPolar(planes[0], planes[1], magnitude, phase);
        
        // Spectral residual computation
        magnitude += cv::Scalar::all(1);
        cv::log(magnitude, magnitude);
        
        cv::Mat avg_magnitude;
        cv::boxFilter(magnitude, avg_magnitude, -1, cv::Size(3, 3));
        cv::Mat spectral_residual = magnitude - avg_magnitude;
        
        cv::Mat mag_exp;
        cv::exp(spectral_residual, mag_exp);
        
        // Inverse FFT
        cv::Mat real_part, imag_part;
        cv::polarToCart(mag_exp, phase, real_part, imag_part);
        
        cv::Mat inverse_planes[] = {real_part, imag_part};
        cv::Mat inverse_complex;
        cv::merge(inverse_planes, 2, inverse_complex);
        
        cv::Mat inverse;
        cv::idft(inverse_complex, inverse);
        cv::split(inverse, inverse_planes);
        
        // Compute magnitude of inverse transform
        cv::Mat channel_saliency;
        cv::magnitude(inverse_planes[0], inverse_planes[1], channel_saliency);
        
        channel_saliency = channel_saliency(cv::Rect(0, 0, image.cols, image.rows));
        saliency_map += channel_saliency;
    }
    
    // Average across channels and smooth
    saliency_map /= 3.0;
    cv::GaussianBlur(saliency_map, saliency_map, cv::Size(11, 11), 0);
    cv::normalize(saliency_map, saliency_map, 0, 1, cv::NORM_MINMAX);
    
    return saliency_map;
}

cv::Mat computeGraphBasedSaliency(const cv::Mat& image) {
    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    lab.convertTo(lab, CV_32F);
    
    cv::Scalar mean = cv::mean(lab);
    cv::Mat saliency = cv::Mat::zeros(image.size(), CV_32F);
    
    // Compute distance from mean color
    for (int i = 0; i < lab.rows; i++) {
        for (int j = 0; j < lab.cols; j++) {
            cv::Vec3f pixel = lab.at<cv::Vec3f>(i, j);
            float dist = 0;
            for (int c = 0; c < 3; c++) {
                float diff = pixel[c] - mean[c];
                dist += diff * diff;
            }
            saliency.at<float>(i, j) = std::sqrt(dist);
        }
    }
    
    cv::GaussianBlur(saliency, saliency, cv::Size(15, 15), 0);
    cv::normalize(saliency, saliency, 0, 1, cv::NORM_MINMAX);
    
    return saliency;
}

std::vector<float> extractSaliencyWeightedHistogram(const cv::Mat& image, 
                                                     const cv::Mat& saliency_map,
                                                     int bins_per_channel) {
    int total_bins = bins_per_channel * bins_per_channel * bins_per_channel;
    std::vector<float> histogram(total_bins, 0.0f);
    float total_weight = 0.0f;
    
    // Weight each pixel's contribution by its saliency
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            float weight = saliency_map.at<float>(i, j);
            
            int b_bin = std::min((int)(pixel[0] * bins_per_channel / 256.0), bins_per_channel - 1);
            int g_bin = std::min((int)(pixel[1] * bins_per_channel / 256.0), bins_per_channel - 1);
            int r_bin = std::min((int)(pixel[2] * bins_per_channel / 256.0), bins_per_channel - 1);
            
            int bin_index = r_bin * bins_per_channel * bins_per_channel + 
                          g_bin * bins_per_channel + b_bin;
            
            histogram[bin_index] += weight;
            total_weight += weight;
        }
    }
    
    // Normalize by total weight
    if (total_weight > 0) {
        for (float& h : histogram) h /= total_weight;
    }
    
    return histogram;
}

std::vector<float> extractSaliencyWeightedTexture(const cv::Mat& image,
                                                   const cv::Mat& saliency_map,
                                                   int bins) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    gray.convertTo(gray, CV_32F);
    
    // Compute Sobel gradients
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
    
    cv::Mat magnitude;
    cv::magnitude(grad_x, grad_y, magnitude);
    
    double max_mag;
    cv::minMaxLoc(magnitude, nullptr, &max_mag);
    
    std::vector<float> histogram(bins, 0.0f);
    float bin_width = max_mag / bins;
    float total_weight = 0.0f;
    
    // Weight gradient histogram by saliency
    for (int i = 0; i < magnitude.rows; i++) {
        for (int j = 0; j < magnitude.cols; j++) {
            float mag = magnitude.at<float>(i, j);
            float weight = saliency_map.at<float>(i, j);
            
            int bin_idx = std::min((int)(mag / bin_width), bins - 1);
            histogram[bin_idx] += weight;
            total_weight += weight;
        }
    }
    
    if (total_weight > 0) {
        for (float& h : histogram) h /= total_weight;
    }
    
    return histogram;
}

std::vector<float> extractSaliencyFeature(const cv::Mat& image) {
    cv::Mat saliency = computeSaliencyMap(image);
    
    // Extract saliency-weighted color and texture histograms
    std::vector<float> color_hist = extractSaliencyWeightedHistogram(image, saliency, 8);
    std::vector<float> texture_hist = extractSaliencyWeightedTexture(image, saliency, 16);
    
    // Combine into single feature vector
    std::vector<float> features;
    features.insert(features.end(), color_hist.begin(), color_hist.end());
    features.insert(features.end(), texture_hist.begin(), texture_hist.end());
    
    return features;
}

cv::Mat visualizeSaliency(const cv::Mat& image, const cv::Mat& saliency_map) {
    cv::Mat saliency_8u;
    saliency_map.convertTo(saliency_8u, CV_8U, 255);
    
    // Apply color map for visualization
    cv::Mat saliency_color;
    cv::applyColorMap(saliency_8u, saliency_color, cv::COLORMAP_JET);
    
    // Blend with original image
    cv::Mat result;
    cv::addWeighted(image, 0.6, saliency_color, 0.4, 0, result);
    
    return result;
}