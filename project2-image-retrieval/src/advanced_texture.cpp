/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Image matching using advanced texture features (GLCM, Gabor, Laws)
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include "csv_utils.h"
#include "distance_metrics.h"
#include "advanced_texture.h"

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
        std::cout << "Example: " << argv[0] << " data/olympus/pic.0535.jpg data/olympus 5" << std::endl;
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

    std::cout << "Target image: " << target_path << std::endl;
    std::cout << "Computing advanced texture features (Co-occurrence + Gabor + Laws)..." << std::endl;

    // Extract 233-dimensional texture feature vector
    std::vector<float> target_features = extractAdvancedTextureFeature(target);
    std::cout << "Feature vector size: " << target_features.size() << std::endl;

    std::vector<ImageMatch> matches;
    std::vector<std::string> image_files = getImageFilenames(database_dir);

    int count = 0;
    for (const std::string &img_path : image_files) {
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) continue;

        std::vector<float> features = extractAdvancedTextureFeature(img);
        float dist = euclideanDistance(target_features, features);

        matches.push_back({img_path, dist});

        count++;
        if (count % 100 == 0) {
            std::cout << "Processed " << count << " images..." << std::endl;
        }
    }

    std::sort(matches.begin(), matches.end());

    std::cout << "\n=== Top " << num_results << " matches (Advanced Texture) ===" << std::endl;
    for (int i = 0; i < std::min(num_results, (int)matches.size()); i++) {
        std::cout << (i + 1) << ". " << matches[i].filename 
                  << " (distance: " << matches[i].distance << ")" << std::endl;
    }
    
    return 0;
}