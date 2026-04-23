/*
 * Shamya Haria
 * February 23, 2026
 * Real-time object recognition with multiple visualization modes
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include "processing.h"

int main(int argc, char** argv) {
    std::cout << "╔════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Real-Time 2D Object Recognition System       ║" << std::endl;
    std::cout << "║  CS5330 - Pattern Recognition & CV            ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════╝\n" << std::endl;
    
    std::cout << "Keyboard Controls:" << std::endl;
    std::cout << "  't' - Toggle training/recognition mode" << std::endl;
    std::cout << "  'n' - Save current object (in training mode)" << std::endl;
    std::cout << "  's' - Save screenshot of all views" << std::endl;
    std::cout << "  'q' - Quit\n" << std::endl;
    
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "❌ ERROR: Cannot open webcam!" << std::endl;
        return -1;
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    
    auto database = loadDatabase();
    std::cout << "✓ Loaded " << database.size() << " object categories\n" << std::endl;
    
    int threshold = 80;
    bool trainingMode = false;
    int screenshotCount = 0;
    
    // Create windows
    cv::namedWindow("Original", cv::WINDOW_NORMAL);
    cv::namedWindow("Matching", cv::WINDOW_NORMAL);
    cv::namedWindow("Clean", cv::WINDOW_NORMAL);
    cv::namedWindow("Segmentation", cv::WINDOW_NORMAL);
    
    // Arrange in 2x2 grid
    cv::resizeWindow("Original", 500, 375);
    cv::resizeWindow("Matching", 500, 375);
    cv::resizeWindow("Clean", 500, 375);
    cv::resizeWindow("Segmentation", 500, 375);
    
    cv::moveWindow("Original", 10, 50);
    cv::moveWindow("Matching", 530, 50);
    cv::moveWindow("Clean", 10, 450);
    cv::moveWindow("Segmentation", 530, 450);
    
    std::cout << "✓ System ready! Point camera at white surface.\n" << std::endl;
    std::cout << "💡 TIP: Use good lighting and dark colored objects!\n" << std::endl;
    
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        
        cv::flip(frame, frame, 1);
        
        // Pipeline processing
        cv::Mat binary;
        customThreshold(frame, binary, threshold);
        
        cv::Mat cleaned;
        morphologicalCleanup(binary, cleaned);
        
        cv::Mat labels, stats, centroids;
        int numRegions;
        segmentRegions(cleaned, labels, stats, centroids, numRegions);
        
        // Create visualizations
        cv::Mat originalDisplay = frame.clone();
        cv::Mat matchingDisplay = frame.clone();
        cv::Mat cleanDisplay;
        cv::cvtColor(255 - cleaned, cleanDisplay, cv::COLOR_GRAY2BGR);
        
        // Colored segmentation
        cv::Mat segDisplay = cv::Mat::zeros(frame.size(), CV_8UC3);
        std::vector<cv::Vec3b> colors(numRegions);
        colors[0] = cv::Vec3b(255, 255, 255);  // Background white
        for (int i = 1; i < numRegions; i++) {
            colors[i] = cv::Vec3b(rand() % 200 + 55, rand() % 200 + 55, rand() % 200 + 55);
        }
        for (int r = 0; r < labels.rows; r++) {
            for (int c = 0; c < labels.cols; c++) {
                segDisplay.at<cv::Vec3b>(r, c) = colors[labels.at<int>(r, c)];
            }
        }
        
        // Find valid regions
        std::vector<int> validRegions;
        for (int i = 1; i < numRegions; i++) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            int x = stats.at<int>(i, cv::CC_STAT_LEFT);
            int y = stats.at<int>(i, cv::CC_STAT_TOP);
            int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            if (area > 300 && y + h < frame.rows - 50) {
                validRegions.push_back(i);
            }       
        }
        
        // Process each detected object
        for (int regionId : validRegions) {
            int x = stats.at<int>(regionId, cv::CC_STAT_LEFT);
            int y = stats.at<int>(regionId, cv::CC_STAT_TOP);
            int w = stats.at<int>(regionId, cv::CC_STAT_WIDTH);
            int h = stats.at<int>(regionId, cv::CC_STAT_HEIGHT);
            
            ObjectFeatures features = computeFeatures(labels, stats, centroids, regionId);
            
            std::string label = "Unknown";
            cv::Scalar boxColor = cv::Scalar(0, 255, 255);  // Yellow
            
            if (!trainingMode && !database.empty()) {
                label = classifyObject(features, database);
                boxColor = cv::Scalar(0, 255, 0);  // Green for recognized
            } else if (trainingMode) {
                label = "Press 'n' to save";
                boxColor = cv::Scalar(0, 165, 255);  // Orange
            }
            
            // Draw on MATCHING view (main detection view)
            cv::rectangle(matchingDisplay, cv::Point(x, y), cv::Point(x+w, y+h), boxColor, 3);
            
            int baseline;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 0.8, 2, &baseline);
            
            cv::rectangle(matchingDisplay,
                         cv::Point(x, y - textSize.height - 12),
                         cv::Point(x + textSize.width + 8, y),
                         boxColor, -1);
            
            cv::putText(matchingDisplay, label,
                       cv::Point(x + 4, y - 6),
                       cv::FONT_HERSHEY_DUPLEX, 0.8,
                       cv::Scalar(0, 0, 0), 2);
            
            // Draw centroid and axis on CLEAN view
            cv::circle(cleanDisplay, 
                      cv::Point(features.centroidX, features.centroidY), 
                      5, cv::Scalar(0, 255, 0), -1);
            
            double length = std::max(w, h) * 0.5;
            double dx = length * cos(features.angle);
            double dy = length * sin(features.angle);
            cv::line(cleanDisplay,
                    cv::Point(features.centroidX - dx, features.centroidY - dy),
                    cv::Point(features.centroidX + dx, features.centroidY + dy),
                    cv::Scalar(255, 0, 0), 2);
        }
        
        // Add headers to windows
        cv::putText(originalDisplay, "Original", cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        
        std::string modeText = trainingMode ? "Matching - TRAINING MODE" : "Matching - RECOGNITION MODE";
        cv::putText(matchingDisplay, modeText, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(matchingDisplay, "Objects: " + std::to_string(validRegions.size()), 
                   cv::Point(10, frame.rows - 15),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        
        cv::putText(cleanDisplay, "Clean", cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        
        cv::putText(segDisplay, "Segmentation Visualization", cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        // Display all views
        cv::imshow("Original", originalDisplay);
        cv::imshow("Matching", matchingDisplay);
        cv::imshow("Clean", cleanDisplay);
        cv::imshow("Segmentation", segDisplay);
        
        char key = cv::waitKey(1);
        
        if (key == 'q') {
            break;
        } else if (key == 't') {
            trainingMode = !trainingMode;
            std::cout << (trainingMode ? "🟡 TRAINING mode" : "🟢 RECOGNITION mode") << std::endl;
        } else if (key == 'n' && !validRegions.empty() && trainingMode) {
            int regionId = validRegions[0];
            ObjectFeatures features = computeFeatures(labels, stats, centroids, regionId);
            
            std::cout << "\n📝 Enter object name: ";
            std::string name;
            std::cin >> name;
            
            saveToDatabase(features, name);
            database = loadDatabase();
            
            std::cout << "✓ Saved! Database: " << database.size() << " categories\n" << std::endl;
        } else if (key == 's') {
            std::string fn1 = "screenshots/original_" + std::to_string(screenshotCount) + ".png";
            std::string fn2 = "screenshots/matching_" + std::to_string(screenshotCount) + ".png";
            std::string fn3 = "screenshots/clean_" + std::to_string(screenshotCount) + ".png";
            std::string fn4 = "screenshots/seg_" + std::to_string(screenshotCount) + ".png";
            
            cv::imwrite(fn1, originalDisplay);
            cv::imwrite(fn2, matchingDisplay);
            cv::imwrite(fn3, cleanDisplay);
            cv::imwrite(fn4, segDisplay);
            
            screenshotCount++;
            std::cout << "📸 Saved 4 screenshots!" << std::endl;
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}