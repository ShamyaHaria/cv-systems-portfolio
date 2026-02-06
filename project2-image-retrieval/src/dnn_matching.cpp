/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Image matching using pre-computed ResNet18 DNN embeddings with cosine distance
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include "csv_utils.h"
#include "distance_metrics.h"

struct ImageMatch {
    std::string filename;
    float distance;

    bool operator<(const ImageMatch &other) const {
        return distance < other.distance;
    }
};

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <target_image> <embeddings_csv> <num_results>" << std::endl;
        std::cout << "Example: " << argv[0] << " data/olympus/pic.0893.jpg data/embeddings.csv 5" << std::endl;
        return -1;
    }

    std::string target_path = argv[1];
    std::string embeddings_file = argv[2];
    int num_results = std::atoi(argv[3]);

    std::cout << "Target image: " << target_path << std::endl;
    std::cout << "Loading DNN embeddings from: " << embeddings_file << std::endl;

    // Extract filename from full path for CSV lookup
    size_t last_slash = target_path.find_last_of("/\\");
    std::string target_filename = (last_slash != std::string::npos)
                                      ? target_path.substr(last_slash + 1)
                                      : target_path;

    // Load all 512-dimensional ResNet18 embeddings
    auto database_features = readFeaturesFromCSV(embeddings_file);

    // Find target image's embedding
    std::vector<float> target_features;
    for (const auto &[filename, features] : database_features) {
        if (filename.find(target_filename) != std::string::npos) {
            target_features = features;
            std::cout << "Found target embedding: " << filename << std::endl;
            break;
        }
    }

    if (target_features.empty()) {
        std::cerr << "Error: Could not find target image in embeddings file" << std::endl;
        return -1;
    }

    std::cout << "Computing cosine distances..." << std::endl;

    std::vector<ImageMatch> matches;

    // Compare target with all database embeddings
    for (const auto &[filename, features] : database_features) {
        float dist = cosineDistance(target_features, features);
        matches.push_back({filename, dist});
    }

    std::sort(matches.begin(), matches.end());

    std::cout << "\n=== Top " << num_results << " matches ===" << std::endl;
    for (int i = 0; i < std::min(num_results, (int)matches.size()); i++) {
        std::cout << (i + 1) << ". " << matches[i].filename 
                  << " (distance: " << matches[i].distance << ")" << std::endl;
    }

    return 0;
}