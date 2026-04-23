// Shamya Haria
// Project 4: Task 7 - Harris Corner Feature Detection

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        return -1;
    }

    std::cout << "Controls:" << std::endl;
    std::cout << "  '+' - increase threshold" << std::endl;
    std::cout << "  '-' - decrease threshold" << std::endl;
    std::cout << "  'q' - quit" << std::endl;

    int threshold = 150;
    cv::Mat frame, gray, harris, harris_norm, harris_scaled;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat small;
        cv::resize(frame, small, cv::Size(640, 480));
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_32F);

        cv::cornerHarris(gray, harris, 2, 3, 0.04);
        cv::normalize(harris, harris_norm, 0, 255, cv::NORM_MINMAX, CV_32F);
        cv::convertScaleAbs(harris_norm, harris_scaled);

        // Draw circles on strong corners
        int count = 0;
        for (int r = 0; r < harris_norm.rows; r++) {
            for (int c = 0; c < harris_norm.cols; c++) {
                if ((int)harris_norm.at<float>(r,c) > threshold) {
                    cv::circle(small, cv::Point(c,r), 5, cv::Scalar(0,0,255), 2);
                    count++;
                }
            }
        }

        cv::putText(small, "Threshold: " + std::to_string(threshold),
            cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
        cv::putText(small, "Features: " + std::to_string(count),
            cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);

        cv::imshow("Harris Features", small);

        char key = cv::waitKey(30);
        if (key == '+' && threshold < 250) threshold += 10;
        if (key == '-' && threshold > 10)  threshold -= 10;
        if (key == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}