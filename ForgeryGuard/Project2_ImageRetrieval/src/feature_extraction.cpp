/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Feature extraction implementations for all CBIR matching methods
 */

#include "feature_extraction.h"
#include <iostream>

std::vector<float> extractBaselineFeature(const cv::Mat &image) {
    std::vector<float> feature;

    int rows = image.rows;
    int cols = image.cols;

    // Find center of image
    int center_row = rows / 2;
    int center_col = cols / 2;

    int half_size = 3;

    // Extract 7x7 window around center
    for (int i = center_row - half_size; i <= center_row + half_size; i++) {
        for (int j = center_col - half_size; j <= center_col + half_size; j++) {
            if (i >= 0 && i < rows && j >= 0 && j < cols) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
                feature.push_back(pixel[0]);
                feature.push_back(pixel[1]);
                feature.push_back(pixel[2]);
            }
        }
    }

    return feature;
}

std::vector<float> extractRGChromaticityHistogram(const cv::Mat &image, int r_bins, int g_bins) {
    std::vector<float> histogram(r_bins * g_bins, 0.0f);

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            float B = pixel[0];
            float G = pixel[1];
            float R = pixel[2];

            float sum = R + G + B;

            if (sum < 1e-6) {
                continue;
            }

            // Normalize by intensity to get chromaticity
            float r = R / sum;
            float g = G / sum;

            // Compute bin indices
            int r_bin = std::min((int)(r * r_bins), r_bins - 1);
            int g_bin = std::min((int)(g * g_bins), g_bins - 1);

            int bin_index = r_bin * g_bins + g_bin;
            histogram[bin_index]++;
        }
    }

    return histogram;
}

std::vector<float> extractRGBHistogram(const cv::Mat &image, int bins_per_channel) {
    std::vector<float> histogram(bins_per_channel * bins_per_channel * bins_per_channel, 0.0f);

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);

            // Map pixel values to bins
            int b_bin = std::min((int)(pixel[0] * bins_per_channel / 256.0), bins_per_channel - 1);
            int g_bin = std::min((int)(pixel[1] * bins_per_channel / 256.0), bins_per_channel - 1);
            int r_bin = std::min((int)(pixel[2] * bins_per_channel / 256.0), bins_per_channel - 1);

            // Convert 3D bin coordinates to linear index
            int bin_index = r_bin * bins_per_channel * bins_per_channel +
                            g_bin * bins_per_channel +
                            b_bin;

            histogram[bin_index]++;
        }
    }

    return histogram;
}

std::vector<float> extractMultiRegionHistogram(const cv::Mat &image, int bins_per_channel) {
    std::vector<float> feature;

    int mid_row = image.rows / 2;

    // Extract histogram for top half
    cv::Mat top_half = image(cv::Rect(0, 0, image.cols, mid_row));
    std::vector<float> top_hist = extractRGBHistogram(top_half, bins_per_channel);

    // Extract histogram for bottom half
    cv::Mat bottom_half = image(cv::Rect(0, mid_row, image.cols, image.rows - mid_row));
    std::vector<float> bottom_hist = extractRGBHistogram(bottom_half, bins_per_channel);

    // Concatenate both histograms
    feature.insert(feature.end(), top_hist.begin(), top_hist.end());
    feature.insert(feature.end(), bottom_hist.begin(), bottom_hist.end());

    return feature;
}

cv::Mat computeSobelMagnitude(const cv::Mat &image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    cv::Mat gray_float;
    gray.convertTo(gray_float, CV_32F);

    // Apply Sobel in X and Y directions
    cv::Mat sobel_x, sobel_y;
    cv::Sobel(gray_float, sobel_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray_float, sobel_y, CV_32F, 0, 1, 3);

    // Compute gradient magnitude
    cv::Mat magnitude;
    cv::magnitude(sobel_x, sobel_y, magnitude);

    return magnitude;
}

std::vector<float> computeHistogram(const cv::Mat &image, int bins, float min_val, float max_val) {
    std::vector<float> histogram(bins, 0.0f);

    float range = max_val - min_val;
    float bin_width = range / bins;

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            float value = image.at<float>(i, j);

            // Compute bin index and clamp to valid range
            int bin_index = (int)((value - min_val) / bin_width);
            bin_index = std::max(0, std::min(bins - 1, bin_index));

            histogram[bin_index]++;
        }
    }

    return histogram;
}

std::vector<float> extractGradientMagnitudeHistogram(const cv::Mat &image, int bins) {
    cv::Mat magnitude = computeSobelMagnitude(image);

    // Find maximum magnitude for histogram range
    double max_mag;
    cv::minMaxLoc(magnitude, nullptr, &max_mag);

    return computeHistogram(magnitude, bins, 0.0f, max_mag);
}

std::vector<float> extractColorTextureFeature(const cv::Mat &image, int color_bins, int texture_bins) {
    std::vector<float> feature;

    // Extract RGB color histogram
    std::vector<float> color_hist = extractRGBHistogram(image, color_bins);

    // Extract gradient magnitude histogram
    std::vector<float> texture_hist = extractGradientMagnitudeHistogram(image, texture_bins);

    // Combine into single feature vector
    feature.insert(feature.end(), color_hist.begin(), color_hist.end());
    feature.insert(feature.end(), texture_hist.begin(), texture_hist.end());

    return feature;
}