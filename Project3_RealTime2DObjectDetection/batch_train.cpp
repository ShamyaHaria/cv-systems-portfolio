/*
 * Shamya Haria
 * February 23, 2026
 * Batch training system - automatically labels objects from filenames
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include "processing.h"

namespace fs = std::filesystem;

std::string extractLabel(const std::string &filename) {
    std::string fn = filename;
    std::transform(fn.begin(), fn.end(), fn.begin(), ::tolower);
    
    if (fn.find("triangle") != std::string::npos) return "triangle";
    if (fn.find("hammer") != std::string::npos) return "hammer";
    if (fn.find("allen") != std::string::npos) return "allen_key";
    if (fn.find("screwdriver") != std::string::npos) return "screwdriver";
    if (fn.find("key_fob") != std::string::npos || fn.find("keyfob") != std::string::npos) return "key_fob";
    if (fn.find("pen") != std::string::npos) return "pen";
    if (fn.find("phone") != std::string::npos) return "phone";
    if (fn.find("mouse") != std::string::npos) return "mouse";
    if (fn.find("glove") != std::string::npos) return "glove";
    if (fn.find("bracelet") != std::string::npos) return "bracelet";
    if (fn.find("scissors") != std::string::npos) return "scissors";
    if (fn.find("cd") != std::string::npos) return "cd";
    if (fn.find("star") != std::string::npos) return "star";
    if (fn.find("postit") != std::string::npos || fn.find("post-it") != std::string::npos) return "postit";
    if (fn.find("box") != std::string::npos) return "box";
    
    return "unknown";
}

int main() {
    std::cout << "=== Batch Training System ===" << std::endl;
    
    std::string folder = "data/train/";
    if (!fs::exists(folder)) {
        std::cout << "Creating data/train/ folder..." << std::endl;
        fs::create_directories(folder);
        std::cout << "Please add training images to data/train/ and run again." << std::endl;
        return 0;
    }
    
    std::remove("object_db.csv");
    
    int threshold = 100;
    int trained = 0;
    
    for (const auto &entry : fs::directory_iterator(folder)) {
        std::string filepath = entry.path().string();
        std::string filename = entry.path().filename().string();
        std::string ext = entry.path().extension().string();
        
        if (ext != ".jpg" && ext != ".png" && ext != ".jpeg") continue;
        
        std::string label = extractLabel(filename);
        if (label == "unknown") {
            std::cout << "Skipping: " << filename << " (cannot extract label)" << std::endl;
            continue;
        }
        
        cv::Mat image = cv::imread(filepath);
        if (image.empty()) continue;
        
        cv::Mat binary;
        customThreshold(image, binary, threshold);
        
        cv::Mat cleaned;
        morphologicalCleanup(binary, cleaned);
        
        cv::Mat labels, stats, centroids;
        int numRegions;
        segmentRegions(cleaned, labels, stats, centroids, numRegions);
        
        int largestRegion = -1;
        int maxArea = 0;
        for (int i = 1; i < numRegions; i++) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area > maxArea && area > 300) {
                maxArea = area;
                largestRegion = i;
            }
        }
        
        if (largestRegion != -1) {
            ObjectFeatures features = computeFeatures(labels, stats, centroids, largestRegion);
            saveToDatabase(features, label);
            trained++;
            std::cout << "✓ Trained: " << filename << " → " << label << std::endl;
        }
    }
    
    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "Total images trained: " << trained << std::endl;
    
    return 0;
}