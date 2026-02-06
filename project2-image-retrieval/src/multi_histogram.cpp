/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Multi-region histogram matching using top and bottom spatial histograms
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

    bool operator<(const ImageMatch &other) const {
        return distance < other.distance;
    }
};

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <target_image> <database_directory> <num_results> [feature_file]" << std::endl;
        std::cout << "Example: " << argv[0] << " data/olympus/pic.0274.jpg data/olympus 5" << std::endl;
        return -1;
    }

    std::string target_path = argv[1];
    std::string database_dir = argv[2];
    int num_results = std::atoi(argv[3]);
    std::string feature_file = (argc > 4) ? argv[4] : "";

    cv::Mat target = cv::imread(target_path);
    if (target.empty()) {
        std::cerr << "Error: Could not read target image: " << target_path << std::endl;
        return -1;
    }

    std::cout << "Target image: " << target_path << std::endl;
    std::cout << "Computing multi-histogram matching (top/bottom halves, RGB)..." << std::endl;

    // Extract top and bottom region histograms
    std::vector<float> target_features = extractMultiRegionHistogram(target, 8);

    // Split into separate top and bottom histograms
    int hist_size = target_features.size() / 2;
    std::vector<float> target_top(target_features.begin(), target_features.begin() + hist_size);
    std::vector<float> target_bottom(target_features.begin() + hist_size, target_features.end());

    std::vector<ImageMatch> matches;

    if (!feature_file.empty()) {
        auto database_features = readFeaturesFromCSV(feature_file);

        for (const auto &[filename, features] : database_features) {
            std::vector<float> db_top(features.begin(), features.begin() + hist_size);
            std::vector<float> db_bottom(features.begin() + hist_size, features.end());

            // Compare top and bottom regions separately
            float dist_top = histogramIntersectionDistance(target_top, db_top);
            float dist_bottom = histogramIntersectionDistance(target_bottom, db_bottom);

            // Average distances with equal weights
            float total_dist = 0.5 * dist_top + 0.5 * dist_bottom;

            matches.push_back({filename, total_dist});
        }
    } else {
        std::vector<std::string> image_files = getImageFilenames(database_dir);

        for (const std::string &img_path : image_files) {
            cv::Mat img = cv::imread(img_path);
            if (img.empty()) continue;

            std::vector<float> features = extractMultiRegionHistogram(img, 8);

            std::vector<float> db_top(features.begin(), features.begin() + hist_size);
            std::vector<float> db_bottom(features.begin() + hist_size, features.end());

            float dist_top = histogramIntersectionDistance(target_top, db_top);
            float dist_bottom = histogramIntersectionDistance(target_bottom, db_bottom);

            float total_dist = 0.5 * dist_top + 0.5 * dist_bottom;

            matches.push_back({img_path, total_dist});
        }
    }

    std::sort(matches.begin(), matches.end());

    std::cout << "\n=== Top " << num_results << " matches ===" << std::endl;
    for (int i = 0; i < std::min(num_results, (int)matches.size()); i++) {
        std::cout << (i + 1) << ". " << matches[i].filename 
                  << " (distance: " << matches[i].distance << ")" << std::endl;
    }

    return 0;
}