/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Distance metrics for comparing feature vectors
 */

#ifndef DISTANCE_METRICS_H
#define DISTANCE_METRICS_H

#include <vector>

// Computes sum of squared differences between two feature vectors
float sumSquaredDifference(const std::vector<float>& f1, 
                          const std::vector<float>& f2);

// Computes histogram intersection (returns 0-1, higher means more similar)
float histogramIntersection(const std::vector<float>& h1, 
                           const std::vector<float>& h2);

// Converts histogram intersection to distance metric (smaller means more similar)
float histogramIntersectionDistance(const std::vector<float>& h1, 
                                   const std::vector<float>& h2);

// Computes cosine distance between two vectors (1 - cosine similarity)
float cosineDistance(const std::vector<float>& v1, 
                    const std::vector<float>& v2);

// Computes Euclidean distance between two vectors
float euclideanDistance(const std::vector<float>& v1, 
                       const std::vector<float>& v2);

// Normalizes vector to unit length (L2 normalization)
std::vector<float> normalizeVector(const std::vector<float>& vec);

// Normalizes histogram to sum to 1
std::vector<float> normalizeHistogram(const std::vector<float>& hist);

#endif