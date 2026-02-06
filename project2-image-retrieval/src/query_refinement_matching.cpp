/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Interactive query refinement with user relevance feedback
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include "csv_utils.h"
#include "distance_metrics.h"
#include "feature_extraction.h"
#include "query_refinement.h"

struct ImageMatch {
    std::string filename;
    float distance;
    bool operator<(const ImageMatch& other) const { return distance < other.distance; }
};

// Searches database with given query features
std::vector<ImageMatch> searchDatabase(const std::vector<float>& query, const std::string& dir) {
    std::vector<ImageMatch> matches;
    for (const auto& path : getImageFilenames(dir)) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;
        
        auto color = extractRGBHistogram(img, 8);
        auto texture = extractGradientMagnitudeHistogram(img, 16);
        
        std::vector<float> features;
        features.insert(features.end(), color.begin(), color.end());
        features.insert(features.end(), texture.begin(), texture.end());
        
        float dist = histogramIntersectionDistance(query, features);
        matches.push_back({path, dist});
    }
    std::sort(matches.begin(), matches.end());
    return matches;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <target> <database> <num_results> [iterations]" << std::endl;
        return -1;
    }
    
    std::string target_path = argv[1];
    std::string database_dir = argv[2];
    int num_results = std::atoi(argv[3]);
    int max_iter = (argc > 4) ? std::atoi(argv[4]) : 3;
    
    cv::Mat target = cv::imread(target_path);
    if (target.empty()) return -1;
    
    std::cout << "=== Query Refinement ===" << std::endl;
    std::cout << "Target: " << target_path << std::endl;
    
    // Extract initial query features
    auto color = extractRGBHistogram(target, 8);
    auto texture = extractGradientMagnitudeHistogram(target, 16);
    
    std::vector<float> initial;
    initial.insert(initial.end(), color.begin(), color.end());
    initial.insert(initial.end(), texture.begin(), texture.end());
    
    QueryRefiner refiner(initial);
    
    // Initial search
    std::cout << "\n=== ITERATION 0: Initial ===" << std::endl;
    auto matches = searchDatabase(initial, database_dir);
    
    std::cout << "Top " << num_results << ":" << std::endl;
    for (int i = 0; i < std::min(num_results, (int)matches.size()); i++) {
        std::cout << (i+1) << ". " << matches[i].filename << std::endl;
    }
    
    // Iterative refinement with simulated user feedback
    for (int iter = 1; iter <= max_iter; iter++) {
        std::cout << "\n=== ITERATION " << iter << " ===" << std::endl;
        
        // Simulate user selecting rank 2 match
        int selected_rank = 1;
        std::string selected = matches[selected_rank].filename;
        std::cout << "Feedback: Selected #" << (selected_rank+1) << ": " << selected << std::endl;
        
        // Extract features from selected image
        cv::Mat sel_img = cv::imread(selected);
        auto sel_color = extractRGBHistogram(sel_img, 8);
        auto sel_texture = extractGradientMagnitudeHistogram(sel_img, 16);
        
        std::vector<float> sel_features;
        sel_features.insert(sel_features.end(), sel_color.begin(), sel_color.end());
        sel_features.insert(sel_features.end(), sel_texture.begin(), sel_texture.end());
        
        // Update query with feedback
        refiner.addFeedback(sel_features);
        auto refined = refiner.getRefinedFeatures();
        
        // Re-search with refined query
        matches = searchDatabase(refined, database_dir);
        
        std::cout << "Top " << num_results << " (refined):" << std::endl;
        for (int i = 0; i < std::min(num_results, (int)matches.size()); i++) {
            std::cout << (i+1) << ". " << matches[i].filename << std::endl;
        }
    }
    
    return 0;
}