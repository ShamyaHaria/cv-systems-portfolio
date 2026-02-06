/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Query refinement implementation for iterative relevance feedback
 */

#include "query_refinement.h"
#include <iostream>
#include <algorithm>

std::vector<float> refineQueryFeatures(const std::vector<float>& original,
                                       const std::vector<float>& selected,
                                       float alpha) {
    if (original.size() != selected.size()) return original;
    
    // Blend original and selected features
    std::vector<float> refined(original.size());
    for (size_t i = 0; i < original.size(); i++) {
        refined[i] = alpha * original[i] + (1.0f - alpha) * selected[i];
    }
    return refined;
}

QueryRefiner::QueryRefiner(const std::vector<float>& initial) 
    : current_query_features(initial), iteration(0) {}

void QueryRefiner::addFeedback(const std::vector<float>& selected) {
    feedback_history.push_back(selected);
    iteration++;
    
    // Decrease alpha with iterations to trust user feedback more
    float alpha = 0.7f / iteration;
    alpha = std::max(0.3f, alpha);
    
    current_query_features = refineQueryFeatures(current_query_features, selected, alpha);
}

std::vector<float> QueryRefiner::getRefinedFeatures() const {
    return current_query_features;
}

void QueryRefiner::reset() {
    feedback_history.clear();
    iteration = 0;
}