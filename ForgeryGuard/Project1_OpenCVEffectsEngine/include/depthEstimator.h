/*
 * Shamya Haria
 * CS 5330 - Project 1
 * Date: Jan 26, 2026
 * 
 * depthEstimator.h
 * Wrapper for depth estimation
 */

#ifndef DEPTH_ESTIMATOR_H
#define DEPTH_ESTIMATOR_H

#include <opencv2/opencv.hpp>

int estimateDepth(cv::Mat &src, cv::Mat &dst);

#endif