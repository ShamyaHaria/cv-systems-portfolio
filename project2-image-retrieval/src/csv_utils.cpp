/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: CSV file I/O utilities for feature vector storage and retrieval
 */

#include "csv_utils.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int writeFeatureToCSV(const std::string& csv_filename, 
                      const std::string& image_filename,
                      const std::vector<float>& features,
                      bool append) {
    std::ofstream file;
    
    // Open in append or overwrite mode
    if (append) {
        file.open(csv_filename, std::ios::app);
    } else {
        file.open(csv_filename, std::ios::out);
    }
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open CSV file for writing: " << csv_filename << std::endl;
        return -1;
    }
    
    // Write filename followed by comma-separated features
    file << image_filename;
    for (const float& feature : features) {
        file << "," << feature;
    }
    file << std::endl;
    
    file.close();
    return 0;
}

std::vector<std::pair<std::string, std::vector<float>>> 
readFeaturesFromCSV(const std::string& csv_filename) {
    std::vector<std::pair<std::string, std::vector<float>>> result;
    
    std::ifstream file(csv_filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open CSV file for reading: " << csv_filename << std::endl;
        return result;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string filename;
        std::string value;
        std::vector<float> features;
        
        // Parse filename (first column)
        std::getline(ss, filename, ',');
        
        // Parse all feature values
        while (std::getline(ss, value, ',')) {
            try {
                features.push_back(std::stof(value));
            } catch (const std::exception& e) {
                std::cerr << "Error parsing feature value: " << value << std::endl;
            }
        }
        
        result.push_back({filename, features});
    }
    
    file.close();
    return result;
}

std::vector<float> readFeatureForImage(const std::string& csv_filename,
                                       const std::string& image_filename) {
    auto all_features = readFeaturesFromCSV(csv_filename);
    
    // Search for matching filename
    for (const auto& [fname, features] : all_features) {
        if (fname == image_filename) {
            return features;
        }
    }
    
    std::cerr << "Warning: Image " << image_filename << " not found in CSV" << std::endl;
    return std::vector<float>();
}

std::vector<std::string> getImageFilenames(const std::string& directory) {
    std::vector<std::string> filenames;
    
    try {
        // Iterate through directory entries
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                // Filter for common image extensions
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || 
                    ext == ".bmp" || ext == ".tif" || ext == ".tiff") {
                    filenames.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading directory: " << e.what() << std::endl;
    }
    
    return filenames;
}