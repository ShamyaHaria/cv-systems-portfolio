/*
 * Shamya Haria
 * January 26, 2026
 *
 * CS 5330 - Project 1
 * 
 * depthEstimator.cpp
 * Gradient-based depth estimation using brightness inversion and center weighting.
 */

#include "depthEstimator.h"

/*
 * estimateDepth - Custom depth estimation from single image
 * Combines brightness inversion, contrast enhancement, smoothing, and center-weighted bias to approximate depth.
 */
int estimateDepth(cv::Mat &src, cv::Mat &dst) {

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    dst.create(src.rows, src.cols, CV_8UC1);

    // Invert brightness and enhance contrast
    for (int i = 0; i < gray.rows; i++) {
        unsigned char *grayRow = gray.ptr<unsigned char>(i);
        unsigned char *dstRow = dst.ptr<unsigned char>(i);
        for (int j = 0; j < gray.cols; j++) {

            int value = 255 - grayRow[j];

            // Piecewise contrast adjustment
            value = (value < 128) ? value / 2 : 128 + (value - 128) * 2;
            if (value > 255) value = 255;
            if (value < 0) value = 0;
            dstRow[j] = (unsigned char)value;
        }
    }

    // Smooth depth map with large Gaussian kernel
    cv::GaussianBlur(dst, dst, cv::Size(31, 31), 0);

    // Apply center-weighted bias assuming subject is centered
    int centerX = dst.cols / 2;
    int centerY = dst.rows / 2;
    float maxDist = sqrt(centerX * centerX + centerY * centerY);

    for (int i = 0; i < dst.rows; i++) {
        unsigned char *dstRow = dst.ptr<unsigned char>(i);
        for (int j = 0; j < dst.cols; j++) {
            float dx = j - centerX;
            float dy = i - centerY;
            float dist = sqrt(dx * dx + dy * dy);
            float distFactor = 1.0f - (dist / maxDist) * 0.5f;

            int newValue = dstRow[j] * distFactor;
            if (newValue > 255) newValue = 255;
            dstRow[j] = (unsigned char)newValue;
        }
    }

    return 0;
}