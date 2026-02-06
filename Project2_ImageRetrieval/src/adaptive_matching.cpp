/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Adaptive feature weighting that automatically determines optimal weights based on image characteristics
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include "csv_utils.h"
#include "distance_metrics.h"
#include "feature_extraction.h"
#include "image_analysis.h"

struct ImageMatch {
    std::string filename;
    float distance;
    
    bool operator<(const ImageMatch &other) const {
        return distance < other.distance;
    }
};

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <target_image> <database_directory> <num_results>" << std::endl;
        std::cout << "Example: " << argv[0] << " data/olympus/pic.0164.jpg data/olympus 5" << std::endl;
        return -1;
    }

    std::string target_path = argv[1];
    std::string database_dir = argv[2];
    int num_results = std::atoi(argv[3]);

    cv::Mat target = cv::imread(target_path);
    if (target.empty()) {
        std::cerr << "Error: Could not read target image: " << target_path << std::endl;
        return -1;
    }

    std::cout << "=== Adaptive Feature Weighting System ===" << std::endl;
    std::cout << "Target image: " << target_path << std::endl;

    // Analyze target image characteristics
    std::cout << "\nAnalyzing target image characteristics..." << std::endl;
    ImageCharacteristics chars = analyzeImage(target);

    std::cout << "  Color Variance: " << chars.color_variance << std::endl;
    std::cout << "  Texture Strength: " << chars.texture_strength << std::endl;
    std::cout << "  Spatial Complexity: " << chars.spatial_complexity << std::endl;
    std::cout << "  Brightness Range: " << chars.brightness_range << std::endl;

    // Compute adaptive weights based on characteristics
    FeatureWeights weights = computeAdaptiveWeights(chars);

    std::cout << "\nComputed Adaptive Weights:" << std::endl;
    std::cout << "  Color Weight: " << weights.color_weight << std::endl;
    std::cout << "  Texture Weight: " << weights.texture_weight << std::endl;
    std::cout << "  Spatial Weight: " << weights.spatial_weight << std::endl;

    // Extract target features
    std::cout << "\nExtracting features..." << std::endl;
    std::vector<float> target_color = extractRGBHistogram(target, 8);
    std::vector<float> target_texture = extractGradientMagnitudeHistogram(target, 16);
    std::vector<float> target_spatial = extractMultiRegionHistogram(target, 8);

    std::vector<ImageMatch> matches;
    std::vector<std::string> image_files = getImageFilenames(database_dir);

    std::cout << "Processing database images..." << std::endl;
    int count = 0;
    for (const std::string &img_path : image_files) {
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) continue;

        // Extract database image features
        std::vector<float> db_color = extractRGBHistogram(img, 8);
        std::vector<float> db_texture = extractGradientMagnitudeHistogram(img, 16);
        std::vector<float> db_spatial = extractMultiRegionHistogram(img, 8);

        // Calculate distances for each feature type
        float dist_color = histogramIntersectionDistance(target_color, db_color);
        float dist_texture = histogramIntersectionDistance(target_texture, db_texture);
        float dist_spatial = histogramIntersectionDistance(target_spatial, db_spatial);

        // Combine using adaptive weights
        float total_dist = weights.color_weight * dist_color +
                           weights.texture_weight * dist_texture +
                           weights.spatial_weight * dist_spatial;

        matches.push_back({img_path, total_dist});

        count++;
        if (count % 100 == 0) {
            std::cout << "  Processed " << count << " images..." << std::endl;
        }
    }

    std::sort(matches.begin(), matches.end());

    std::cout << "\n=== Top " << num_results << " matches (Adaptive Weighting) ===" << std::endl;
    for (int i = 0; i < std::min(num_results, (int)matches.size()); i++) {
        std::cout << (i + 1) << ". " << matches[i].filename
                  << " (distance: " << matches[i].distance << ")" << std::endl;
    }

    // Compare with fixed equal weights for reference
    std::cout << "\n=== Comparison with Fixed Equal Weights ===" << std::endl;
    std::vector<ImageMatch> fixed_matches;
    for (const std::string &img_path : image_files) {
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) continue;

        std::vector<float> db_color = extractRGBHistogram(img, 8);
        std::vector<float> db_texture = extractGradientMagnitudeHistogram(img, 16);
        std::vector<float> db_spatial = extractMultiRegionHistogram(img, 8);

        float dist_color = histogramIntersectionDistance(target_color, db_color);
        float dist_texture = histogramIntersectionDistance(target_texture, db_texture);
        float dist_spatial = histogramIntersectionDistance(target_spatial, db_spatial);

        float total_dist = 0.33 * dist_color + 0.33 * dist_texture + 0.34 * dist_spatial;
        fixed_matches.push_back({img_path, total_dist});
    }
    std::sort(fixed_matches.begin(), fixed_matches.end());

    std::cout << "Top 3 with fixed weights:" << std::endl;
    for (int i = 0; i < std::min(3, (int)fixed_matches.size()); i++) {
        std::cout << (i + 1) << ". " << fixed_matches[i].filename
                  << " (distance: " << fixed_matches[i].distance << ")" << std::endl;
    }

    return 0;
}