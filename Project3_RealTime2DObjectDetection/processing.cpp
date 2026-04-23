/*
 * Shamya Haria
 * February 23, 2026
 * Implementation of real-time object recognition processing functions
 */

#include "processing.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>

// Custom threshold implementation (from scratch requirement)
void customThreshold(const cv::Mat &src, cv::Mat &dst, int thresholdValue) {
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    
    dst = cv::Mat::zeros(gray.size(), CV_8UC1);
    
    for (int row = 0; row < gray.rows; row++) {
        for (int col = 0; col < gray.cols; col++) {
            uchar pixel = gray.at<uchar>(row, col);
            dst.at<uchar>(row, col) = (pixel < thresholdValue) ? 255 : 0;
        }
    }
}

// Clean binary image with morphological operations
void morphologicalCleanup(const cv::Mat &src, cv::Mat &dst) {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat closed;
    cv::morphologyEx(src, closed, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(closed, dst, cv::MORPH_OPEN, kernel);
}

// Segment image into connected regions
void segmentRegions(const cv::Mat &src, cv::Mat &labels, cv::Mat &stats, 
                   cv::Mat &centroids, int &numRegions) {
    numRegions = cv::connectedComponentsWithStats(src, labels, stats, centroids, 8, CV_32S);
}

// Extract rotation and scale invariant features
ObjectFeatures computeFeatures(const cv::Mat &labels, const cv::Mat &stats, 
                               const cv::Mat &centroids, int regionId) {
    ObjectFeatures features;
    
    int x = stats.at<int>(regionId, cv::CC_STAT_LEFT);
    int y = stats.at<int>(regionId, cv::CC_STAT_TOP);
    int w = stats.at<int>(regionId, cv::CC_STAT_WIDTH);
    int h = stats.at<int>(regionId, cv::CC_STAT_HEIGHT);
    int area = stats.at<int>(regionId, cv::CC_STAT_AREA);
    
    features.centroidX = centroids.at<double>(regionId, 0);
    features.centroidY = centroids.at<double>(regionId, 1);
    features.width = w;
    features.height = h;
    features.aspectRatio = (double)h / w;
    features.percentFilled = (double)area / (w * h);
    
    cv::Mat regionMask = (labels == regionId);
    cv::Moments m = cv::moments(regionMask, true);
    
    features.angle = 0.5 * atan2(2 * m.mu11, m.mu20 - m.mu02);
    cv::HuMoments(m, features.huMoments);
    
    return features;
}

// Save features to CSV database
void saveToDatabase(const ObjectFeatures &features, const std::string &label) {
    std::ofstream file("object_db.csv", std::ios::app);
    
    file << label << ","
         << features.percentFilled << ","
         << features.aspectRatio << ","
         << features.huMoments[0] << ","
         << features.huMoments[1] << ","
         << features.huMoments[2] << ","
         << features.huMoments[3] << ","
         << features.huMoments[4] << ","
         << features.huMoments[5] << ","
         << features.huMoments[6] << "\n";
    
    file.close();
}

// Load database from CSV
std::map<std::string, std::vector<ObjectFeatures>> loadDatabase() {
    std::map<std::string, std::vector<ObjectFeatures>> db;
    std::ifstream file("object_db.csv");
    
    if (!file.is_open()) {
        return db;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string label;
        ObjectFeatures features;
        
        std::getline(ss, label, ',');
        ss >> features.percentFilled; ss.ignore();
        ss >> features.aspectRatio; ss.ignore();
        ss >> features.huMoments[0]; ss.ignore();
        ss >> features.huMoments[1]; ss.ignore();
        ss >> features.huMoments[2]; ss.ignore();
        ss >> features.huMoments[3]; ss.ignore();
        ss >> features.huMoments[4]; ss.ignore();
        ss >> features.huMoments[5]; ss.ignore();
        ss >> features.huMoments[6];
        
        features.label = label;
        db[label].push_back(features);
    }
    
    file.close();
    return db;
}

// Classify using nearest neighbor
std::string classifyObject(const ObjectFeatures &unknown, 
                          const std::map<std::string, std::vector<ObjectFeatures>> &db) {
    if (db.empty()) {
        return "No Training Data";
    }
    
    std::string bestLabel = "Unknown";
    double minDistance = std::numeric_limits<double>::max();
    
    for (const auto &entry : db) {
        for (const auto &known : entry.second) {
            double d1 = (unknown.percentFilled - known.percentFilled) / 0.2;
            double d2 = (unknown.aspectRatio - known.aspectRatio) / 1.0;
            double d3 = (unknown.huMoments[0] - known.huMoments[0]) / 0.1;
            double d4 = (unknown.huMoments[1] - known.huMoments[1]) / 0.01;
            
            double distance = sqrt(d1*d1 + d2*d2 + d3*d3 + d4*d4);
            
            if (distance < minDistance) {
                minDistance = distance;
                bestLabel = entry.first;
            }
        }
    }
    
    return bestLabel;
}