/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Object detection using MobileNet-SSD for 20 PASCAL VOC categories
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include "csv_utils.h"

struct Detection {
    std::string filename;
    float confidence;
    bool operator<(const Detection& other) const { return confidence > other.confidence; }
};

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <database> <num_results> <object_name>" << std::endl;
        std::cout << "Objects: bottle, chair, person, car, bicycle, bird, etc." << std::endl;
        return -1;
    }
    
    std::string db = argv[1];
    int num = std::atoi(argv[2]);
    std::string target = argv[3];
    
    std::cout << "Loading MobileNet-SSD model..." << std::endl;
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(
        "../models/mobilenet_ssd.prototxt",
        "../models/mobilenet_ssd.caffemodel"
    );
    
    // Load class names from file
    std::vector<std::string> classes;
    std::ifstream file("../models/ssd_classes.txt");
    std::string line;
    while (std::getline(file, line)) classes.push_back(line);
    
    std::vector<Detection> results;
    auto files = getImageFilenames(db);
    
    int count = 0;
    for (const auto& path : files) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;
        
        // Preprocess image to 300x300 blob
        cv::Mat blob = cv::dnn::blobFromImage(img, 0.007843, cv::Size(300, 300), 
                                               cv::Scalar(127.5, 127.5, 127.5), false);
        net.setInput(blob);
        cv::Mat detection = net.forward();
        
        // Parse detection matrix
        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        
        float max_conf = 0.0f;
        for (int i = 0; i < detectionMat.rows; i++) {
            float conf = detectionMat.at<float>(i, 2);
            if (conf > 0.3) {
                int classId = (int)detectionMat.at<float>(i, 1);
                if (classId < classes.size() && classes[classId] == target) {
                    max_conf = std::max(max_conf, conf);
                }
            }
        }
        
        if (max_conf > 0) results.push_back({path, max_conf});
        
        if (++count % 50 == 0) std::cout << "  " << count << " images..." << std::endl;
    }
    
    std::sort(results.begin(), results.end());
    
    std::cout << "\n=== Top " << num << " " << target << " detections ===" << std::endl;
    for (int i = 0; i < std::min(num, (int)results.size()); i++) {
        std::cout << (i+1) << ". " << results[i].filename << " (" << results[i].confidence << ")\n";
    }
    
    if (results.empty()) std::cout << "No " << target << " found.\n";
    
    return 0;
}