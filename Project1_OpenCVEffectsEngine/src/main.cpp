#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(img, cv::Point(100, 100), cv::Point(540, 380), cv::Scalar(0, 255, 0), 3);
    cv::putText(img, "OpenCV Works!", cv::Point(150, 240), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 2);
    std::string path = "../data/test_output.jpg";
    if (cv::imwrite(path, img)) {
        std::cout << "Saved: " << path << std::endl;
    }
    std::cout << "OpenCV test complete!" << std::endl;
    return 0;
}
