/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Custom retrieval combining spatial histograms, texture, and DNN embeddings with 40-30-30 weighting
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include "csv_utils.h"
#include "distance_metrics.h"
#include "feature_extraction.h"

struct ImageMatch {
    std::string filename;
    float distance;
    
    bool operator<(const ImageMatch& other) const {
        return distance < other.distance;
    }
};

// Combines multi-region histograms with texture features
std::vector<float> extractCustomFeature(const cv::Mat& image) {
    std::vector<float> feature;
    
    std::vector<float> spatial_hist = extractMultiRegionHistogram(image, 8);
    std::vector<float> texture = extractGradientMagnitudeHistogram(image, 16);
    
    feature.insert(feature.end(), spatial_hist.begin(), spatial_hist.end());
    feature.insert(feature.end(), texture.begin(), texture.end());
    
    return feature;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <target_image> <database_directory> <num_results> [embeddings_csv]" << std::endl;
        std::cout << "Example: " << argv[0] << " data/olympus/pic.1072.jpg data/olympus 5 data/embeddings.csv" << std::endl;
        return -1;
    }
    
    std::string target_path = argv[1];
    std::string database_dir = argv[2];
    int num_results = std::atoi(argv[3]);
    std::string embeddings_file = (argc > 4) ? argv[4] : "";
    
    cv::Mat target = cv::imread(target_path);
    if (target.empty()) {
        std::cerr << "Error: Could not read target image: " << target_path << std::endl;
        return -1;
    }
    
    std::cout << "Target image: " << target_path << std::endl;
    std::cout << "Computing custom features (spatial + texture + DNN)..." << std::endl;
    
    std::vector<float> target_custom = extractCustomFeature(target);
    
    // Extract filename for DNN lookup
    size_t last_slash = target_path.find_last_of("/\\");
    std::string target_filename = (last_slash != std::string::npos) 
                                  ? target_path.substr(last_slash + 1) 
                                  : target_path;
    
    // Load DNN embeddings if provided
    std::vector<float> target_dnn;
    std::vector<std::pair<std::string, std::vector<float>>> dnn_embeddings;
    
    if (!embeddings_file.empty()) {
        dnn_embeddings = readFeaturesFromCSV(embeddings_file);
        for (const auto& [filename, features] : dnn_embeddings) {
            if (filename.find(target_filename) != std::string::npos) {
                target_dnn = features;
                break;
            }
        }
    }
    
    std::vector<ImageMatch> matches;
    std::vector<std::string> image_files = getImageFilenames(database_dir);
    
    for (const std::string& img_path : image_files) {
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) continue;
        
        std::vector<float> db_custom = extractCustomFeature(img);
        
        // Split into spatial and texture components
        int spatial_size = 8 * 8 * 8 * 2;
        std::vector<float> target_spatial(target_custom.begin(), target_custom.begin() + spatial_size);
        std::vector<float> target_texture(target_custom.begin() + spatial_size, target_custom.end());
        std::vector<float> db_spatial(db_custom.begin(), db_custom.begin() + spatial_size);
        std::vector<float> db_texture(db_custom.begin() + spatial_size, db_custom.end());
        
        // Calculate distances
        float dist_spatial = histogramIntersectionDistance(target_spatial, db_spatial);
        float dist_texture = histogramIntersectionDistance(target_texture, db_texture);
        
        // Weighted combination (40% spatial, 30% texture, 30% DNN)
        float total_dist = 0.4 * dist_spatial + 0.3 * dist_texture;
        
        // Add DNN component if available
        if (!target_dnn.empty()) {
            size_t img_last_slash = img_path.find_last_of("/\\");
            std::string img_filename = (img_last_slash != std::string::npos) 
                                      ? img_path.substr(img_last_slash + 1) 
                                      : img_path;
            
            for (const auto& [filename, dnn_feat] : dnn_embeddings) {
                if (filename.find(img_filename) != std::string::npos) {
                    float dist_dnn = cosineDistance(target_dnn, dnn_feat);
                    total_dist += 0.3 * dist_dnn;
                    break;
                }
            }
        }
        
        matches.push_back({img_path, total_dist});
    }
    
    std::sort(matches.begin(), matches.end());
    
    std::cout << "\n=== Top " << num_results << " matches ===" << std::endl;
    for (int i = 0; i < std::min(num_results, (int)matches.size()); i++) {
        std::cout << (i + 1) << ". " << matches[i].filename 
                  << " (distance: " << matches[i].distance << ")" << std::endl;
    }
    
    // Display least similar images
    std::cout << "\n=== Least similar images ===" << std::endl;
    int start = std::max(0, (int)matches.size() - 5);
    for (int i = start; i < matches.size(); i++) {
        std::cout << (i + 1) << ". " << matches[i].filename 
                  << " (distance: " << matches[i].distance << ")" << std::endl;
    }
    
    return 0;
}