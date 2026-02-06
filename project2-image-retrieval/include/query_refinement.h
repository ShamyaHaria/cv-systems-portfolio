/*
 * Author: Shamya Haria
 * Date: February 5, 2026
 * Purpose: Query refinement with relevance feedback for iterative search improvement
 */

#ifndef QUERY_REFINEMENT_H
#define QUERY_REFINEMENT_H

#include <opencv2/opencv.hpp>
#include <vector>

// Blends original query features with user-selected match features using alpha parameter
std::vector<float> refineQueryFeatures(const std::vector<float>& original,const std::vector<float>& selected,float alpha = 0.7);

// Manages iterative query refinement with user feedback
class QueryRefiner {
private:
    std::vector<float> current_query_features;
    std::vector<std::vector<float>> feedback_history;
    int iteration;
    
public:
    QueryRefiner(const std::vector<float>& initial);
    
    // Incorporates user feedback from selected match into refined query
    void addFeedback(const std::vector<float>& selected);
    
    // Returns current refined feature vector
    std::vector<float> getRefinedFeatures() const;
    
    int getIteration() const { return iteration; }
    
    // Resets to initial query state
    void reset();
};

#endif