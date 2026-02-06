/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Image matching using saliency-weighted color and texture features
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include "csv_utils.h"
#include "distance_metrics.h"
#include "saliency_features.h"

struct ImageMatch {
    std::string filename;
    float distance;
    bool operator<(const ImageMatch& other) const { return distance < other.distance; }
};

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <target> <database> <num_results> [save_vis]" << std::endl;
        return -1;
    }
    
    std::string target_path = argv[1];
    std::string database_dir = argv[2];
    int num_results = std::atoi(argv[3]);
    bool save_vis = (argc > 4 && std::string(argv[4]) == "true");
    
    cv::Mat target = cv::imread(target_path);
    if (target.empty()) {
        std::cerr << "Error: Could not read target image" << std::endl;
        return -1;
    }
    
    std::cout << "=== Saliency-Based Matching ===" << std::endl;
    std::cout << "Target: " << target_path << std::endl;
    
    // Compute saliency map for target
    std::cout << "\nComputing saliency map..." << std::endl;
    cv::Mat saliency = computeSaliencyMap(target);
    
    // Save visualization if requested
    if (save_vis) {
        cv::Mat vis = visualizeSaliency(target, saliency);
        cv::imwrite("../results/extensions/target_saliency.jpg", vis);
        std::cout << "Saved saliency visualization" << std::endl;
    }
    
    std::cout << "Extracting features..." << std::endl;
    std::vector<float> target_features = extractSaliencyFeature(target);
    std::cout << "Feature size: " << target_features.size() << std::endl;
    
    std::vector<ImageMatch> matches;
    std::vector<std::string> files = getImageFilenames(database_dir);
    
    std::cout << "\nProcessing database..." << std::endl;
    int count = 0;
    for (const auto& path : files) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;
        
        auto features = extractSaliencyFeature(img);
        float dist = histogramIntersectionDistance(target_features, features);
        matches.push_back({path, dist});
        
        if (++count % 100 == 0) std::cout << "  " << count << " images..." << std::endl;
    }
    
    std::sort(matches.begin(), matches.end());
    
    std::cout << "\n=== Top " << num_results << " matches ===" << std::endl;
    for (int i = 0; i < std::min(num_results, (int)matches.size()); i++) {
        std::cout << (i+1) << ". " << matches[i].filename 
                  << " (distance: " << matches[i].distance << ")" << std::endl;
    }
    
    return 0;
}