/*
 * Shamya Haria
 * January 26, 2026
 *
 * CS 5330 - Project 1
 *
 * vidDisplay.cpp
 * Real-time video capture and display with interactive filter selection.
 * Main program that captures from webcam and applies effects based on keyboard input.
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include "filters.h"
#include "depthEstimator.h"

int main(int argc, char *argv[])
{
    cv::VideoCapture *capdev;

    // Open default camera
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened())
    {
        printf("ERROR: Unable to open video device\n");
        return -1;
    }

    // Set camera resolution
    capdev->set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capdev->set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Warm up camera by capturing dummy frames
    std::cout << "Initializing camera, please wait..." << std::endl;
    cv::Mat dummy;
    for (int i = 0; i < 30; i++)
    {
        *capdev >> dummy;
    }
    std::cout << "Camera ready!" << std::endl;

    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Camera opened successfully\n");
    printf("Resolution: %d x %d\n", refS.width, refS.height);

    // Display keyboard controls
    std::cout << "\n=== Video Display Controls ===" << std::endl;
    std::cout << "q - Quit" << std::endl;
    std::cout << "s - Save current frame" << std::endl;
    std::cout << "h - grayscale (Custom)" << std::endl;
    std::cout << "g - grayscale (OpenCV)" << std::endl;
    std::cout << "p - sepia tone" << std::endl;
    std::cout << "b - blur (5x5)" << std::endl;
    std::cout << "x - Sobel X (vertical edges)" << std::endl;
    std::cout << "y - Sobel Y (horizontal edges)" << std::endl;
    std::cout << "m - gradient magnitude" << std::endl;
    std::cout << "l - blur quantize (cartoon effect)" << std::endl;
    std::cout << "f - face detection" << std::endl;
    std::cout << "d - depth map" << std::endl;
    std::cout << "t - depth focus (portrait mode)" << std::endl;
    std::cout << "k - sketch mode" << std::endl;
    std::cout << "i - spotlight face" << std::endl;
    std::cout << "n - glitch effect" << std::endl;
    std::cout << "c - color pop effect (cycles through R/G/B)" << std::endl;
    std::cout << "o - Spider-Man mask" << std::endl;
    std::cout << "z - Run blur timing test" << std::endl;
    std::cout << "\nStarting video stream..." << std::endl;

    cv::namedWindow("Video", 1);
    cv::Mat frame;

    int frameCount = 0;
    int savedCount = 0;

    // Mode flags for each filter
    bool grayscaleMode = false;
    bool customGrayscaleMode = false;
    bool sepiaMode = false;
    bool blurMode = false;
    bool sobelXMode = false;
    bool sobelYMode = false;
    bool magnitudeMode = false;
    bool blurQuantizeMode = false;
    bool faceDetectMode = false;
    bool depthMode = false;
    bool depthFocusMode = false;
    bool sketchModeActive = false;
    bool spotlightMode = false;
    bool glitchMode = false;
    bool colorPopMode = false;
    int colorChannel = 2;
    bool spidermanMode = false;

    // Main capture and display loop
    for (;;)
    {
        *capdev >> frame;

        if (frame.empty())
        {
            printf("ERROR: Frame is empty\n");
            break;
        }

        frameCount++;

        cv::Mat displayFrame;

        // Apply selected filter
        if (grayscaleMode)
        {
            cv::cvtColor(frame, displayFrame, cv::COLOR_BGR2GRAY);
            cv::cvtColor(displayFrame, displayFrame, cv::COLOR_GRAY2BGR);
        }
        else if (customGrayscaleMode)
        {
            greyscale(frame, displayFrame);
        }
        else if (sepiaMode)
        {
            sepia(frame, displayFrame);
        }
        else if (blurMode)
        {
            blur5x5_2(frame, displayFrame);
        }
        else if (sobelXMode)
        {
            cv::Mat sobelX;
            sobelX3x3(frame, sobelX);
            cv::convertScaleAbs(sobelX, displayFrame);
        }
        else if (sobelYMode)
        {
            cv::Mat sobelY;
            sobelY3x3(frame, sobelY);
            cv::convertScaleAbs(sobelY, displayFrame);
        }
        else if (magnitudeMode)
        {
            cv::Mat sobelX, sobelY;
            sobelX3x3(frame, sobelX);
            sobelY3x3(frame, sobelY);
            magnitude(sobelX, sobelY, displayFrame);
        }
        else if (blurQuantizeMode)
        {
            blurQuantize(frame, displayFrame, 10);
        }
        else if (faceDetectMode)
        {
            displayFrame = frame.clone();
            std::vector<cv::Rect> faces;
            detectFaces(frame, faces);
            for (size_t i = 0; i < faces.size(); i++)
            {
                cv::rectangle(displayFrame, faces[i], cv::Scalar(0, 255, 0), 3);
            }
        }
        else if (depthMode)
        {
            cv::Mat depth;
            estimateDepth(frame, depth);
            cv::applyColorMap(depth, displayFrame, cv::COLORMAP_TURBO);
        }
        else if (depthFocusMode)
        {
            cv::Mat depth;
            estimateDepth(frame, depth);
            depthFocusEffect(frame, depth, displayFrame);
        }
        else if (sketchModeActive)
        {
            sketchFilter(frame, displayFrame);
        }
        else if (spotlightMode)
        {
            displayFrame = frame.clone();
            std::vector<cv::Rect> faces;
            detectFaces(frame, faces);
            spotlightFace(frame, faces, displayFrame);
        }
        else if (glitchMode)
        {
            glitchEffect(frame, displayFrame);
        }
        else if (colorPopMode)
        {
            colorPop(frame, displayFrame, colorChannel);
        }
        else if (spidermanMode)
        {
            displayFrame = frame.clone();
            std::vector<cv::Rect> faces;
            detectFaces(frame, faces);
            spidermanMask(frame, faces, displayFrame);
        }
        else
        {
            displayFrame = frame.clone();
        }

        // Overlay frame count
        std::string frameText = "Frame: " + std::to_string(frameCount);
        cv::putText(displayFrame, frameText, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // Overlay current mode
        std::string modeText = "Mode: Color";
        if (grayscaleMode)
            modeText = "Mode: Grayscale (OpenCV)";
        if (customGrayscaleMode)
            modeText = "Mode: Grayscale (Custom)";
        if (sepiaMode)
            modeText = "Mode: Sepia Tone";
        if (blurMode)
            modeText = "Mode: Blur (5x5)";
        if (sobelXMode)
            modeText = "Mode: Sobel X (Vertical Edges)";
        if (sobelYMode)
            modeText = "Mode: Sobel Y (Horizontal Edges)";
        if (magnitudeMode)
            modeText = "Mode: Gradient Magnitude";
        if (blurQuantizeMode)
            modeText = "Mode: Blur Quantize";
        if (faceDetectMode)
            modeText = "Mode: Face Detection";
        if (depthMode)
            modeText = "Mode: Depth Map";
        if (depthFocusMode)
            modeText = "Mode: Depth Focus";
        if (sketchModeActive)
            modeText = "Mode: Sketch";
        if (spotlightMode)
            modeText = "Mode: Spotlight Face";
        if (glitchMode)
            modeText = "Mode: Glitch Effect";
        if (colorPopMode) {
            if (colorChannel == 2) modeText = "Mode: Color Pop (Red)";
            else if (colorChannel == 1) modeText = "Mode: Color Pop (Green)";
            else modeText = "Mode: Color Pop (Blue)";
        }
        if (spidermanMode)
            modeText = "Mode: Spider-Man Mask";
        cv::putText(displayFrame, modeText, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Video", displayFrame);

        // Check for keyboard input
        int key = cv::waitKey(30);

        if (key >= 0)
        {
            key = key & 0xFF;
        }

        // Handle keyboard commands
        if (key == 'q' || key == 27)
        {
            std::cout << "\nQuitting..." << std::endl;
            break;
        }
        else if (key == 's')
        {
            savedCount++;
            std::string filename = "../data/frame_" + std::to_string(savedCount) + ".jpg";
            cv::imwrite(filename, displayFrame);
            std::cout << "Saved: " << filename << std::endl;
        }
        else if (key == 'g')
        {
            grayscaleMode = !grayscaleMode;
            if (grayscaleMode)
            {
                customGrayscaleMode = false;
                sepiaMode = false;
                blurMode = false;
                sobelXMode = false;
                sobelYMode = false;
                magnitudeMode = false;
                blurQuantizeMode = false;
                faceDetectMode = false;
                depthMode = false;
                depthFocusMode = false;
                sketchModeActive = false;
                spotlightMode = false;
                glitchMode = false;
                colorPopMode = false;
                spidermanMode = false;
                std::cout << "OpenCV grayscale: ON" << std::endl;
            }
            else
            {
                std::cout << "OpenCV grayscale: OFF" << std::endl;
            }
        }
        else if (key == 'h')
        {
            customGrayscaleMode = !customGrayscaleMode;
            if (customGrayscaleMode)
            {
                grayscaleMode = false;
                sepiaMode = false;
                blurMode = false;
                sobelXMode = false;
                sobelYMode = false;
                magnitudeMode = false;
                blurQuantizeMode = false;
                faceDetectMode = false;
                depthFocusMode = false;
                depthMode = false;
                sketchModeActive = false;
                spotlightMode = false;
                glitchMode = false;
                colorPopMode = false;
                spidermanMode = false;
                std::cout << "Custom grayscale: ON" << std::endl;
            }
            else
            {
                std::cout << "Custom grayscale: OFF" << std::endl;
            }
        }
        else if (key == 'p')
        {
            sepiaMode = !sepiaMode;
            if (sepiaMode)
            {
                grayscaleMode = false;
                customGrayscaleMode = false;
                blurMode = false;
                sobelXMode = false;
                sobelYMode = false;
                magnitudeMode = false;
                blurQuantizeMode = false;
                faceDetectMode = false;
                depthMode = false;
                depthFocusMode = false;
                sketchModeActive = false;
                spotlightMode = false;
                glitchMode = false;
                colorPopMode = false;
                spidermanMode = false;
                std::cout << "Sepia tone: ON" << std::endl;
            }
            else
            {
                std::cout << "Sepia tone: OFF" << std::endl;
            }
        }
        else if (key == 'b')
        {
            blurMode = !blurMode;
            if (blurMode)
            {
                grayscaleMode = false;
                customGrayscaleMode = false;
                sepiaMode = false;
                sobelXMode = false;
                sobelYMode = false;
                magnitudeMode = false;
                blurQuantizeMode = false;
                faceDetectMode = false;
                depthMode = false;
                depthFocusMode = false;
                sketchModeActive = false;
                spotlightMode = false;
                glitchMode = false;
                colorPopMode = false;
                spidermanMode = false;
                std::cout << "Blur: ON" << std::endl;
            }
            else
            {
                std::cout << "Blur: OFF" << std::endl;
            }
        }
        else if (key == 'x')
        {
            sobelXMode = !sobelXMode;
            if (sobelXMode)
            {
                grayscaleMode = false;
                customGrayscaleMode = false;
                sepiaMode = false;
                blurMode = false;
                sobelYMode = false;
                magnitudeMode = false;
                blurQuantizeMode = false;
                faceDetectMode = false;
                depthFocusMode = false;
                depthMode = false;
                sketchModeActive = false;
                spotlightMode = false;
                glitchMode = false;
                colorPopMode = false;
                spidermanMode = false;
                std::cout << "Sobel X: ON" << std::endl;
            }
            else
            {
                std::cout << "Sobel X: OFF" << std::endl;
            }
        }
        else if (key == 'y')
        {
            sobelYMode = !sobelYMode;
            if (sobelYMode)
            {
                grayscaleMode = false;
                customGrayscaleMode = false;
                sepiaMode = false;
                blurMode = false;
                sobelXMode = false;
                magnitudeMode = false;
                blurQuantizeMode = false;
                faceDetectMode = false;
                depthFocusMode = false;
                depthMode = false;
                sketchModeActive = false;
                spotlightMode = false;
                glitchMode = false;
                colorPopMode = false;
                spidermanMode = false;
                std::cout << "Sobel Y: ON" << std::endl;
            }
            else
            {
                std::cout << "Sobel Y: OFF" << std::endl;
            }
        }
        else if (key == 'm')
        {
            magnitudeMode = !magnitudeMode;
            if (magnitudeMode)
            {
                grayscaleMode = false;
                customGrayscaleMode = false;
                sepiaMode = false;
                blurMode = false;
                sobelXMode = false;
                sobelYMode = false;
                blurQuantizeMode = false;
                faceDetectMode = false;
                depthFocusMode = false;
                depthMode = false;
                sketchModeActive = false;
                spotlightMode = false;
                glitchMode = false;
                colorPopMode = false;
                spidermanMode = false;
                std::cout << "Gradient magnitude: ON" << std::endl;
            }
            else
            {
                std::cout << "Gradient magnitude: OFF" << std::endl;
            }
        }
        else if (key == 'l')
        {
            blurQuantizeMode = !blurQuantizeMode;
            if (blurQuantizeMode)
            {
                grayscaleMode = false;
                customGrayscaleMode = false;
                sepiaMode = false;
                blurMode = false;
                sobelXMode = false;
                sobelYMode = false;
                magnitudeMode = false;
                faceDetectMode = false;
                depthFocusMode = false;
                depthMode = false;
                sketchModeActive = false;
                spotlightMode = false;
                glitchMode = false;
                colorPopMode = false;
                spidermanMode = false;
                std::cout << "Blur quantize: ON" << std::endl;
            }
            else
            {
                std::cout << "Blur quantize: OFF" << std::endl;
            }
        }
        else if (key == 'f')
        {
            faceDetectMode = !faceDetectMode;
            if (faceDetectMode)
            {
                grayscaleMode = false;
                customGrayscaleMode = false;
                sepiaMode = false;
                blurMode = false;
                sobelXMode = false;
                sobelYMode = false;
                magnitudeMode = false;
                blurQuantizeMode = false;
                depthFocusMode = false;
                depthMode = false;
                sketchModeActive = false;
                spotlightMode = false;
                glitchMode = false;
                colorPopMode = false;
                spidermanMode = false;
                std::cout << "Face detection: ON" << std::endl;
            }
            else
            {
                std::cout << "Face detection: OFF" << std::endl;
            }
        }
        else if (key == 'd')
        {
            depthMode = !depthMode;
            if (depthMode)
            {
                grayscaleMode = false;
                customGrayscaleMode = false;
                sepiaMode = false;
                blurMode = false;
                sobelXMode = false;
                sobelYMode = false;
                magnitudeMode = false;
                blurQuantizeMode = false;
                faceDetectMode = false;
                depthFocusMode = false;
                sketchModeActive = false;
                spotlightMode = false;
                glitchMode = false;
                colorPopMode = false;
                spidermanMode = false;
                std::cout << "Depth map: ON" << std::endl;
            }
            else
            {
                std::cout << "Depth map: OFF" << std::endl;
            }
        }
        else if (key == 't')
        {
            depthFocusMode = !depthFocusMode;
            if (depthFocusMode)
            {
                grayscaleMode = false;
                customGrayscaleMode = false;
                sepiaMode = false;
                blurMode = false;
                sobelXMode = false;
                sobelYMode = false;
                magnitudeMode = false;
                blurQuantizeMode = false;
                faceDetectMode = false;
                depthMode = false;
                sketchModeActive = false;
                spotlightMode = false;
                glitchMode = false;
                colorPopMode = false;
                spidermanMode = false;
                std::cout << "Depth focus: ON" << std::endl;
            }
            else
            {
                std::cout << "Depth focus: OFF" << std::endl;
            }
        }
        else if (key == 'k')
        {
            sketchModeActive = !sketchModeActive;
            if (sketchModeActive)
            {
                grayscaleMode = false;
                customGrayscaleMode = false;
                sepiaMode = false;
                blurMode = false;
                sobelXMode = false;
                sobelYMode = false;
                magnitudeMode = false;
                blurQuantizeMode = false;
                faceDetectMode = false;
                depthMode = false;
                depthFocusMode = false;
                spotlightMode = false;
                glitchMode = false;
                colorPopMode = false;
                spidermanMode = false;
                std::cout << "Sketch mode: ON" << std::endl;
            }
            else
            {
                std::cout << "Sketch mode: OFF" << std::endl;
            }
        }
        else if (key == 'i')
        {
            spotlightMode = !spotlightMode;
            if (spotlightMode)
            {
                grayscaleMode = false;
                customGrayscaleMode = false;
                sepiaMode = false;
                blurMode = false;
                sobelXMode = false;
                sobelYMode = false;
                magnitudeMode = false;
                blurQuantizeMode = false;
                faceDetectMode = false;
                depthMode = false;
                depthFocusMode = false;
                sketchModeActive = false;
                glitchMode = false;
                colorPopMode = false;
                spidermanMode = false;
                std::cout << "Spotlight face: ON" << std::endl;
            }
            else
            {
                std::cout << "Spotlight face: OFF" << std::endl;
            }
        }
        else if (key == 'n')
        {
            glitchMode = !glitchMode;
            if (glitchMode)
            {
                grayscaleMode = false;
                customGrayscaleMode = false;
                sepiaMode = false;
                blurMode = false;
                sobelXMode = false;
                sobelYMode = false;
                magnitudeMode = false;
                blurQuantizeMode = false;
                faceDetectMode = false;
                depthMode = false;
                depthFocusMode = false;
                sketchModeActive = false;
                spotlightMode = false;
                colorPopMode = false;
                spidermanMode = false;
                std::cout << "Glitch effect: ON" << std::endl;
            }
            else
            {
                std::cout << "Glitch effect: OFF" << std::endl;
            }
        }
        else if (key == 'o')
        {
            spidermanMode = !spidermanMode;
            if (spidermanMode)
            {
                grayscaleMode = false;
                customGrayscaleMode = false;
                sepiaMode = false;
                blurMode = false;
                sobelXMode = false;
                sobelYMode = false;
                magnitudeMode = false;
                blurQuantizeMode = false;
                faceDetectMode = false;
                depthMode = false;
                depthFocusMode = false;
                sketchModeActive = false;
                spotlightMode = false;
                glitchMode = false;
                colorPopMode = false;
                std::cout << "Spider-Man mask: ON" << std::endl;
            }
            else
            {
                std::cout << "Spider-Man mask: OFF" << std::endl;
            }
        }
        else if (key == 'z')
        {
            std::cout << "\nRunning blur timing test..." << std::endl;
            testBlurTiming(frame);
        }
        else if (key == 'c')
        {
            if (!colorPopMode) {
                colorPopMode = true;
                colorChannel = 2;
                grayscaleMode = false;
                customGrayscaleMode = false;
                sepiaMode = false;
                blurMode = false;
                sobelXMode = false;
                sobelYMode = false;
                magnitudeMode = false;
                blurQuantizeMode = false;
                faceDetectMode = false;
                depthMode = false;
                depthFocusMode = false;
                sketchModeActive = false;
                spotlightMode = false;
                glitchMode = false;
                spidermanMode = false;
                std::cout << "Color pop: ON (Red channel)" << std::endl;
            } else {
                // Cycle through colors
                if (colorChannel == 2) {
                    colorChannel = 1;
                    std::cout << "Color pop: Green channel" << std::endl;
                } else if (colorChannel == 1) {
                    colorChannel = 0;
                    std::cout << "Color pop: Blue channel" << std::endl;
                } else {
                    colorPopMode = false;
                    colorChannel = 2;
                    std::cout << "Color pop: OFF" << std::endl;
                }
            }
        }
    }

    delete capdev;
    cv::destroyAllWindows();

    std::cout << "Total frames processed: " << frameCount << std::endl;
    std::cout << "Images saved: " << savedCount << std::endl;

    return 0;
}