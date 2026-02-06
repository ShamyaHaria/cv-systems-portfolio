/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: CSV file utilities for reading and writing feature vectors
 */

#ifndef CSV_UTILS_H
#define CSV_UTILS_H

#include <string>
#include <vector>

// Writes feature vector to CSV file in format: filename,feature1,feature2,...
// Returns 0 on success, -1 on failure
int writeFeatureToCSV(const std::string& csv_filename, 
                      const std::string& image_filename,
                      const std::vector<float>& features,
                      bool append = true);

// Reads all feature vectors from CSV file
// Returns vector of (filename, feature_vector) pairs
std::vector<std::pair<std::string, std::vector<float>>> 
readFeaturesFromCSV(const std::string& csv_filename);

// Reads feature vector for a specific image from CSV file
std::vector<float> readFeatureForImage(const std::string& csv_filename,
                                       const std::string& image_filename);

// Gets all image filenames from a directory (jpg, png, bmp)
std::vector<std::string> getImageFilenames(const std::string& directory);

#endif