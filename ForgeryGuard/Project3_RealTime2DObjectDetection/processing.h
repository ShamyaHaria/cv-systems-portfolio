/*
 * Shamya Haria
 * February 23, 2026
 * Header file for real-time object recognition processing functions
 */

#ifndef PROCESSING_H
#define PROCESSING_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>

// Thresholding (from scratch)
void customThreshold(const cv::Mat &src, cv::Mat &dst, int thresholdValue);

// Morphological filtering
void morphologicalCleanup(const cv::Mat &src, cv::Mat &dst);

// Connected components
void segmentRegions(const cv::Mat &src, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids, int &numRegions);

// Feature structure
struct ObjectFeatures {
    double centroidX, centroidY;
    double angle;
    double width, height;
    double percentFilled;
    double aspectRatio;
    double huMoments[7];
    std::string label;
};

// Compute features for a region
ObjectFeatures computeFeatures(const cv::Mat &labels, const cv::Mat &stats, 
                               const cv::Mat &centroids, int regionId);

// Database operations
void saveToDatabase(const ObjectFeatures &features, const std::string &label);
std::map<std::string, std::vector<ObjectFeatures>> loadDatabase();

// Classification
std::string classifyObject(const ObjectFeatures &features, 
                          const std::map<std::string, std::vector<ObjectFeatures>> &db);

#endif