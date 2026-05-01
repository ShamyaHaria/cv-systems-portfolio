/*
 * Shamya Haria
 * January 26, 2026
 *
 * CS 5330 - Project 1
 * 
 * filters.cpp
 * Implementation of image filter functions including basic filters,
 * edge detection, face-based effects, and creative filters.
 */

#include "filters.h"
#include <chrono>

/*
 * greyscale - Custom grayscale conversion using inverted red channel
 * Creates unique artistic effect with emphasized cool tones and higher contrast.
 */
int greyscale(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.rows, src.cols, CV_8UC3);

    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            unsigned char red = srcRow[j][2];
            unsigned char gray = 255 - red;

            dstRow[j][0] = gray;
            dstRow[j][1] = gray;
            dstRow[j][2] = gray;
        }
    }
    return 0;
}

/*
 * sepia - Apply sepia tone filter for vintage photograph effect
 * Uses standard transformation matrix with original RGB values, clamps output to prevent overflow.
 */
int sepia(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.rows, src.cols, CV_8UC3);

    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            unsigned char blue = srcRow[j][0];
            unsigned char green = srcRow[j][1];
            unsigned char red = srcRow[j][2];

            float newBlue = 0.272 * red + 0.534 * green + 0.131 * blue;
            float newGreen = 0.349 * red + 0.686 * green + 0.168 * blue;
            float newRed = 0.393 * red + 0.769 * green + 0.189 * blue;

            dstRow[j][0] = (newBlue > 255) ? 255 : (unsigned char)newBlue;
            dstRow[j][1] = (newGreen > 255) ? 255 : (unsigned char)newGreen;
            dstRow[j][2] = (newRed > 255) ? 255 : (unsigned char)newRed;
        }
    }
    return 0;
}

/*
 * blur5x5_1 - Naive 5x5 Gaussian blur implementation
 * Straightforward implementation using full kernel with at() accessor, performs 25 multiplications per pixel.
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {

    dst = src.clone();

    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    for (int i = 2; i < src.rows - 2; i++) {
        for (int j = 2; j < src.cols - 2; j++) {

            int blueSum = 0, greenSum = 0, redSum = 0;

            for (int ki = -2; ki <= 2; ki++) {
                for (int kj = -2; kj <= 2; kj++) {

                    cv::Vec3b pixel = src.at<cv::Vec3b>(i + ki, j + kj);
                    int weight = kernel[ki + 2][kj + 2];

                    blueSum += pixel[0] * weight;
                    greenSum += pixel[1] * weight;
                    redSum += pixel[2] * weight;
                }
            }

            dst.at<cv::Vec3b>(i, j)[0] = blueSum / 256;
            dst.at<cv::Vec3b>(i, j)[1] = greenSum / 256;
            dst.at<cv::Vec3b>(i, j)[2] = redSum / 256;
        }
    }

    return 0;
}

/*
 * blur5x5_2 - Optimized 5x5 Gaussian blur using separable filters
 * Decomposes 2D convolution into horizontal and vertical passes, uses pointer arithmetic for faster memory access.
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    cv::Mat temp;
    temp.create(src.rows, src.cols, CV_8UC3);

    int filter[5] = {1, 2, 4, 2, 1};

    // Horizontal pass
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *tempRow = temp.ptr<cv::Vec3b>(i);

        for (int j = 2; j < src.cols - 2; j++) {
            int blueSum = 0, greenSum = 0, redSum = 0;

            for (int k = -2; k <= 2; k++) {
                blueSum += srcRow[j + k][0] * filter[k + 2];
                greenSum += srcRow[j + k][1] * filter[k + 2];
                redSum += srcRow[j + k][2] * filter[k + 2];
            }

            tempRow[j][0] = blueSum / 16;
            tempRow[j][1] = greenSum / 16;
            tempRow[j][2] = redSum / 16;
        }

        // Copy boundary pixels
        if (i < 2 || i >= src.rows - 2) {
            for (int j = 0; j < src.cols; j++) {
                tempRow[j] = srcRow[j];
            }
        } else {
            tempRow[0] = srcRow[0];
            tempRow[1] = srcRow[1];
            tempRow[src.cols - 2] = srcRow[src.cols - 2];
            tempRow[src.cols - 1] = srcRow[src.cols - 1];
        }
    }

    dst.create(src.rows, src.cols, CV_8UC3);

    // Vertical pass
    for (int i = 2; i < temp.rows - 2; i++) {
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < temp.cols; j++) {
            int blueSum = 0, greenSum = 0, redSum = 0;

            for (int k = -2; k <= 2; k++) {
                cv::Vec3b *tempRow = temp.ptr<cv::Vec3b>(i + k);
                blueSum += tempRow[j][0] * filter[k + 2];
                greenSum += tempRow[j][1] * filter[k + 2];
                redSum += tempRow[j][2] * filter[k + 2];
            }

            dstRow[j][0] = blueSum / 16;
            dstRow[j][1] = greenSum / 16;
            dstRow[j][2] = redSum / 16;
        }
    }

    for (int i = 0; i < 2; i++) {
        src.row(i).copyTo(dst.row(i));
        src.row(src.rows - 1 - i).copyTo(dst.row(dst.rows - 1 - i));
    }

    return 0;
}

/*
 * sobelX3x3 - Sobel X filter for vertical edge detection
 * Detects vertical edges using separable filters, output uses signed 16-bit integers to preserve gradient polarity.
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {

    cv::Mat temp;
    temp.create(src.rows, src.cols, CV_16SC3);
    dst.create(src.rows, src.cols, CV_16SC3);

    // Horizontal derivative
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                tempRow[j][c] = -srcRow[j-1][c] + srcRow[j+1][c];
            }
        }

        tempRow[0] = cv::Vec3s(0, 0, 0);
        tempRow[src.cols - 1] = cv::Vec3s(0, 0, 0);
    }

    // Vertical smoothing
    for (int i = 1; i < temp.rows - 1; i++) {
        cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);

        for (int j = 0; j < temp.cols; j++) {
            cv::Vec3s *tempRowPrev = temp.ptr<cv::Vec3s>(i - 1);
            cv::Vec3s *tempRowCurr = temp.ptr<cv::Vec3s>(i);
            cv::Vec3s *tempRowNext = temp.ptr<cv::Vec3s>(i + 1);

            for (int c = 0; c < 3; c++) {
                dstRow[j][c] = tempRowPrev[j][c] + 2 * tempRowCurr[j][c] + tempRowNext[j][c];
            }
        }
    }

    dst.row(0).setTo(cv::Scalar(0, 0, 0));
    dst.row(dst.rows - 1).setTo(cv::Scalar(0, 0, 0));

    return 0;
}

/*
 * sobelY3x3 - Sobel Y filter for horizontal edge detection
 * Detects horizontal edges using separable filters, output uses signed 16-bit integers.
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {

    cv::Mat temp;
    temp.create(src.rows, src.cols, CV_16SC3);
    dst.create(src.rows, src.cols, CV_16SC3);

    // Horizontal smoothing
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                tempRow[j][c] = srcRow[j-1][c] + 2 * srcRow[j][c] + srcRow[j+1][c];
            }
        }

        tempRow[0] = cv::Vec3s(0, 0, 0);
        tempRow[src.cols - 1] = cv::Vec3s(0, 0, 0);
    }

    // Vertical derivative
    for (int i = 1; i < temp.rows - 1; i++) {
        cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);

        for (int j = 0; j < temp.cols; j++) {
            cv::Vec3s *tempRowPrev = temp.ptr<cv::Vec3s>(i - 1);
            cv::Vec3s *tempRowNext = temp.ptr<cv::Vec3s>(i + 1);

            for (int c = 0; c < 3; c++) {
                dstRow[j][c] = -tempRowPrev[j][c] + tempRowNext[j][c];
            }
        }
    }

    dst.row(0).setTo(cv::Scalar(0, 0, 0));
    dst.row(dst.rows - 1).setTo(cv::Scalar(0, 0, 0));

    return 0;
}

/*
 * magnitude - Compute gradient magnitude from Sobel X and Y outputs
 * Combines horizontal and vertical gradients using Euclidean distance, computed per channel with clamping to 255.
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    dst.create(sx.rows, sx.cols, CV_8UC3);
    
    for (int i = 0; i < sx.rows; i++) {
        cv::Vec3s *sxRow = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s *syRow = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < sx.cols; j++) {
            for (int c = 0; c < 3; c++) {
                float gx = sxRow[j][c];
                float gy = syRow[j][c];

                float mag = sqrt(gx * gx + gy * gy);
                if (mag > 255) mag = 255;

                dstRow[j][c] = (unsigned char)mag;
            }
        }
    }
    return 0;
}

/*
 * blurQuantize - Cartoon effect combining blur, quantization, and edge darkening
 * Creates comic book style by blurring, posterizing colors into discrete levels, and darkening strong edges.
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    cv::Mat blurred;
    blur5x5_2(src, blurred);
    
    cv::Mat quantized;
    quantized.create(src.rows, src.cols, CV_8UC3);

    int bucketSize = 255 / levels;

    // Color quantization
    for (int i = 0; i < blurred.rows; i++) {
        cv::Vec3b *blurredRow = blurred.ptr<cv::Vec3b>(i);
        cv::Vec3b *quantizedRow = quantized.ptr<cv::Vec3b>(i);

        for (int j = 0; j < blurred.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int value = blurredRow[j][c];
                int q = (value / bucketSize) * bucketSize;
                quantizedRow[j][c] = (unsigned char)q;
            }
        }
    }

    // Edge detection on original
    cv::Mat sobelX, sobelY;
    sobelX3x3(src, sobelX);
    sobelY3x3(src, sobelY);

    cv::Mat edges;
    magnitude(sobelX, sobelY, edges);
    
    dst.create(src.rows, src.cols, CV_8UC3);

    // Combine quantized colors with edge outlines
    for (int i = 0; i < quantized.rows; i++) {
        cv::Vec3b *quantizedRow = quantized.ptr<cv::Vec3b>(i);
        cv::Vec3b *edgesRow = edges.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < quantized.cols; j++) {
            int edgeStrength = edgesRow[j][0];

            if (edgeStrength > 80) {
                dstRow[j][0] = 0;
                dstRow[j][1] = 0;
                dstRow[j][2] = 0;
            } else {
                for (int c = 0; c < 3; c++) {
                    dstRow[j][c] = quantizedRow[j][c];
                }
            }
        }
    }

    return 0;
}

/*
 * detectFaces - Detect faces using Haar cascade classifier
 * Loads cascade on first call, applies histogram equalization for robust detection under varying lighting.
 */
int detectFaces(cv::Mat &frame, std::vector<cv::Rect> &faces) {
    static cv::CascadeClassifier face_cascade;
    static bool loaded = false;

    if (!loaded) {
        if (!face_cascade.load("../data/haarcascade_frontalface_alt2.xml")) {
            std::cout << "Error loading face cascade!" << std::endl;
            return -1;
        }
        loaded = true;
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));

    return 0;
}

/*
 * depthFocusEffect - Portrait mode effect with depth-based selective blur
 * Creates shallow depth-of-field by blending sharp and blurred versions based on depth map values.
 */
int depthFocusEffect(cv::Mat &src, cv::Mat &depth, cv::Mat &dst) {

    cv::Mat blurred;
    blur5x5_2(src, blurred);
    blur5x5_2(blurred, blurred);

    dst = src.clone();

    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *blurredRow = blurred.ptr<cv::Vec3b>(i);
        unsigned char *depthRow = depth.ptr<unsigned char>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            float depthValue = depthRow[j] / 255.0f;
            float blurAmount = 1.0f - depthValue;

            for (int c = 0; c < 3; c++) {
                dstRow[j][c] = srcRow[j][c] * (1.0f - blurAmount) + 
                              blurredRow[j][c] * blurAmount;
            }
        }
    }
    return 0;
}

/*
 * sketchFilter - Pencil sketch effect using edge detection
 * Creates hand-drawn appearance by inverting edges with contrast enhancement and subtle paper tinting.
 */
int sketchFilter(cv::Mat &src, cv::Mat &dst) {

    cv::Mat sobelX, sobelY, edges;
    sobelX3x3(src, sobelX);
    sobelY3x3(src, sobelY);
    magnitude(sobelX, sobelY, edges);

    cv::Mat gray;
    cv::cvtColor(edges, gray, cv::COLOR_BGR2GRAY);

    dst.create(src.rows, src.cols, CV_8UC3);

    for (int i = 0; i < gray.rows; i++) {
        unsigned char *grayRow = gray.ptr<unsigned char>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < gray.cols; j++) {
            int inverted = 255 - grayRow[j];

            int value = (inverted > 200) ? 255 : inverted * 1.2;
            if (value > 255) value = 255;

            dstRow[j][0] = value * 0.9;
            dstRow[j][1] = value * 0.95;
            dstRow[j][2] = value;
        }
    }

    return 0;
}

/*
 * spotlightFace - Dramatic lighting effect emphasizing detected faces
 * Creates theatrical spotlight with radial brightness masks and quadratic falloff, handles multiple faces.
 */
int spotlightFace(cv::Mat &src, std::vector<cv::Rect> &faces, cv::Mat &dst) {
    dst = src.clone();

    if (faces.empty()) {
        dst = dst * 0.3;
        return 0;
    }

    cv::Mat mask = cv::Mat::zeros(src.rows, src.cols, CV_32FC1);

    for (size_t f = 0; f < faces.size(); f++) {
        cv::Rect face = faces[f];

        int expansion = 80;
        cv::Rect expanded(
            std::max(0, face.x - expansion),
            std::max(0, face.y - expansion),
            std::min(src.cols - face.x + expansion, face.width + 2 * expansion),
            std::min(src.rows - face.y + expansion, face.height + 2 * expansion)
        );

        cv::Point2f center(face.x + face.width / 2.0f, face.y + face.height / 2.0f);
        float maxDist = sqrt(expanded.width * expanded.width + expanded.height * expanded.height) / 2.0f;

        for (int i = expanded.y; i < expanded.y + expanded.height && i < mask.rows; i++) {
            float *maskRow = mask.ptr<float>(i);
            for (int j = expanded.x; j < expanded.x + expanded.width && j < mask.cols; j++) {
                float dx = j - center.x;
                float dy = i - center.y;
                float dist = sqrt(dx * dx + dy * dy);
                
                float brightness = 1.0f - (dist / maxDist);
                if (brightness < 0) brightness = 0;
                brightness = brightness * brightness;

                if (brightness > maskRow[j]) {
                    maskRow[j] = brightness;
                }
            }
        }
    }

    for (int i = 0; i < dst.rows; i++) {
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        float *maskRow = mask.ptr<float>(i);

        for (int j = 0; j < dst.cols; j++) {
            float brightness = 0.2f + maskRow[j] * 0.8f;

            for (int c = 0; c < 3; c++) {
                dstRow[j][c] = dstRow[j][c] * brightness;
            }
        }
    }

    return 0;
}

/*
 * glitchEffect - Analog TV interference simulation
 * Creates retro aesthetic with grayscale conversion, monochrome noise overlay, and scanlines.
 */
int glitchEffect(cv::Mat &src, cv::Mat &dst) {

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray, dst, cv::COLOR_GRAY2BGR);

    cv::Mat noise(src.rows, src.cols, CV_8UC3);
    cv::randu(noise, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    cv::Mat grayNoise;
    cv::cvtColor(noise, grayNoise, cv::COLOR_BGR2GRAY);
    cv::cvtColor(grayNoise, noise, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        cv::Vec3b *noiseRow = noise.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            for (int c = 0; c < 3; c++) {
                dstRow[j][c] = dstRow[j][c] * 0.5 + noiseRow[j][c] * 0.5;
            }
        }
    }

    // Add scanlines
    for (int i = 0; i < dst.rows; i += 2) {
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < dst.cols; j++) {
            for (int c = 0; c < 3; c++) {
                dstRow[j][c] *= 0.7;
            }
        }
    }

    return 0;
}

/*
 * colorPop - Selective color isolation effect
 * Isolates target color using saturation and channel dominance, converts non-matching pixels to grayscale.
 */
int colorPop(cv::Mat &src, cv::Mat &dst, int channelToKeep) {
    dst.create(src.rows, src.cols, CV_8UC3);
    
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        
        for (int j = 0; j < src.cols; j++) {
            unsigned char b = srcRow[j][0];
            unsigned char g = srcRow[j][1];
            unsigned char r = srcRow[j][2];
            
            unsigned char gray = 0.299 * r + 0.587 * g + 0.114 * b;
            
            int maxVal = std::max({r, g, b});
            int minVal = std::min({r, g, b});
            int saturation = maxVal - minVal;
            
            bool isTargetColor = false;
            
            if (channelToKeep == 2) {
                // Red detection with skin tone exclusion
                bool isSkinTone = (r > 140 && g > 85 && b > 70) ||
                                  (r > 100 && g > 60 && b > 40 && r - g < 50);
                
                if (!isSkinTone &&
                    r > 100 &&
                    r > g + 45 &&
                    r > b + 55 &&
                    saturation > 70 &&
                    maxVal < 210 &&
                    b < 120) {
                    isTargetColor = true;
                }
            } else if (channelToKeep == 1) {
                if (g > 60 && 
                    g > r + 15 && 
                    g > b + 15 && 
                    saturation > 30) {
                    isTargetColor = true;
                }
            } else {
                if (b > 50 &&
                    b > r &&
                    b > g &&
                    saturation > 20) {
                    isTargetColor = true;
                }
            }
            
            if (isTargetColor) {
                dstRow[j][0] = b;
                dstRow[j][1] = g;
                dstRow[j][2] = r;
            } else {
                dstRow[j][0] = gray;
                dstRow[j][1] = gray;
                dstRow[j][2] = gray;
            }
        }
    }
    return 0;
}

/*
 * spidermanMask - Overlay Spider-Man mask on detected faces
 * Dynamically scales and positions mask based on estimated head boundaries with alpha blending.
 */
int spidermanMask(cv::Mat &src, std::vector<cv::Rect> &faces, cv::Mat &dst) {
    dst = src.clone();

    if (faces.empty()) {
        return 0;
    }

    static cv::Mat maskImage;
    static bool maskLoaded = false;

    if (!maskLoaded) {
        maskImage = cv::imread("../data/spiderman_mask.png", cv::IMREAD_UNCHANGED);
        if (maskImage.empty()) {
            maskImage = cv::imread("../data/spiderman_mask.png");
        }
        if (maskImage.empty()) {
            std::cout << "Warning: Could not load spiderman_mask.png" << std::endl;
            return -1;
        }
        maskLoaded = true;
    }

    for (size_t f = 0; f < faces.size(); f++) {
        cv::Rect face = faces[f];

        int headWidth = face.width * 1.5;
        int headHeight = face.height * 1.8;

        cv::Mat resizedMask;
        cv::resize(maskImage, resizedMask, cv::Size(headWidth, headHeight));

        int xPos = face.x - (headWidth - face.width) / 2;
        int yPos = face.y - face.height * 0.4;

        for (int y = 0; y < resizedMask.rows; y++) {
            int dstY = yPos + y;
            if (dstY < 0 || dstY >= dst.rows) continue;

            cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(dstY);

            for (int x = 0; x < resizedMask.cols; x++) {
                int dstX = xPos + x;
                if (dstX < 0 || dstX >= dst.cols) continue;

                if (resizedMask.channels() == 4) {
                    cv::Vec4b maskPixel = resizedMask.at<cv::Vec4b>(y, x);
                    float alpha = maskPixel[3] / 255.0f;

                    if (alpha > 0.1) {
                        for (int c = 0; c < 3; c++) {
                            dstRow[dstX][c] = dstRow[dstX][c] * (1 - alpha) + maskPixel[c] * alpha;
                        }
                    }
                } else {
                    cv::Vec3b maskPixel = resizedMask.at<cv::Vec3b>(y, x);

                    if (maskPixel[0] + maskPixel[1] + maskPixel[2] > 30) {
                        for (int c = 0; c < 3; c++) {
                            dstRow[dstX][c] = dstRow[dstX][c] * 0.1 + maskPixel[c] * 0.9;
                        }
                    }
                }
            }
        }
    }

    return 0;
}

/*
 * testBlurTiming - Performance comparison of blur implementations
 * Runs both implementations 100 times and reports average execution time and speedup factor.
 */
void testBlurTiming(cv::Mat &testImage) {
    cv::Mat dst1, dst2;
    
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        blur5x5_1(testImage, dst1);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        blur5x5_2(testImage, dst2);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    
    double avgTime1 = duration1.count() / 100.0 / 1000.0;
    double avgTime2 = duration2.count() / 100.0 / 1000.0;
    double speedup = avgTime1 / avgTime2;
    
    std::cout << "\n=== Blur Timing Results ===" << std::endl;
    std::cout << "Image size: " << testImage.cols << "x" << testImage.rows << std::endl;
    std::cout << "blur5x5_1 (naive): " << avgTime1 << " ms" << std::endl;
    std::cout << "blur5x5_2 (separable): " << avgTime2 << " ms" << std::endl;
    std::cout << "Speedup: " << speedup << "x faster" << std::endl;
    std::cout << "=========================\n" << std::endl;
}