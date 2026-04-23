// Shamya Haria
// Project 4: Calibration and Augmented Reality
// Task 1: Detect corners, Task 2: Save frames, Task 3: Calibrate camera

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

const int BOARD_COLS = 9;
const int BOARD_ROWS = 6;

std::vector<cv::Vec3f> buildPointSet() {
    std::vector<cv::Vec3f> point_set;
    for (int r = 0; r < BOARD_ROWS; r++) {
        for (int c = 0; c < BOARD_COLS; c++) {
            point_set.push_back(cv::Vec3f(c, -r, 0));
        }
    }
    return point_set;
}

void runCalibration(std::vector<std::vector<cv::Point2f>> &corner_list,
                    std::vector<std::vector<cv::Vec3f>> &point_list,
                    cv::Size imageSize) {
    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    camera_matrix.at<double>(0,2) = imageSize.width / 2.0;
    camera_matrix.at<double>(1,2) = imageSize.height / 2.0;

    cv::Mat dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);

    std::cout << "Camera matrix BEFORE:\n" << camera_matrix << std::endl;
    std::cout << "Distortion BEFORE:\n" << dist_coeffs << std::endl;

    std::vector<cv::Mat> rvecs, tvecs;

    double error = cv::calibrateCamera(
        point_list, corner_list,
        imageSize,
        camera_matrix, dist_coeffs,
        rvecs, tvecs,
        cv::CALIB_FIX_ASPECT_RATIO
    );

    std::cout << "\nCamera matrix AFTER:\n" << camera_matrix << std::endl;
    std::cout << "Distortion AFTER:\n" << dist_coeffs << std::endl;
    std::cout << "Reprojection error: " << error << " pixels" << std::endl;

    cv::FileStorage fs("calibration.yml", cv::FileStorage::WRITE);
    fs << "camera_matrix" << camera_matrix;
    fs << "dist_coeffs" << dist_coeffs;
    fs.release();

    std::cout << "Calibration saved to calibration.yml" << std::endl;
}

int main() {
    std::vector<std::vector<cv::Point2f>> corner_list;
    std::vector<std::vector<cv::Vec3f>> point_list;
    cv::Size imageSize;

    // Load saved calibration images from results/
    std::vector<std::string> filenames = {
        "results/Tilt1.png", "results/Tilt2.png",
        "results/flat.png", "results/closeup.png", "results/faraway.png",
        "results/corner1.png", "results/corner2.png",
        "results/corner3.png", "results/corner4.png"
    };

    std::cout << "Loading saved calibration images..." << std::endl;

    for (auto &fname : filenames) {
        cv::Mat img = cv::imread(fname);
        if (img.empty()) {
            std::cout << "Could not load: " << fname << std::endl;
            continue;
        }

        imageSize = img.size();
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corner_set;
        bool found = cv::findChessboardCorners(
            gray, cv::Size(BOARD_COLS, BOARD_ROWS), corner_set,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK
        );

        if (found) {
            cv::cornerSubPix(gray, corner_set, cv::Size(11,11), cv::Size(-1,-1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));
            corner_list.push_back(corner_set);
            point_list.push_back(buildPointSet());
            std::cout << "Loaded: " << fname << std::endl;
        } else {
            std::cout << "No corners found in: " << fname << std::endl;
        }
    }

    std::cout << "\nTotal frames loaded: " << corner_list.size() << std::endl;

    if ((int)corner_list.size() < 5) {
        std::cout << "Not enough frames for calibration!" << std::endl;
        return -1;
    }

    runCalibration(corner_list, point_list, imageSize);

    // Also start live camera for Task 1 demo
    cv::VideoCapture cap(1);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        return -1;
    }

    std::cout << "\nLive camera started. Press 's' to save frame, 'q' to quit." << std::endl;

    cv::Mat frame, gray;
    std::vector<cv::Point2f> corner_set;
    bool last_found = false;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        bool found = cv::findChessboardCorners(
            gray, cv::Size(BOARD_COLS, BOARD_ROWS), corner_set,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK
        );

        if (found) {
            cv::cornerSubPix(gray, corner_set, cv::Size(11,11), cv::Size(-1,-1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));
            cv::drawChessboardCorners(frame, cv::Size(BOARD_COLS, BOARD_ROWS), corner_set, found);
            std::cout << "Corners: " << corner_set.size()
                      << "  First: (" << corner_set[0].x << ", " << corner_set[0].y << ")" << std::endl;
            last_found = true;
        } else {
            last_found = false;
        }

        cv::putText(frame, "Saved: " + std::to_string(corner_list.size()),
            cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);

        if (!last_found)
            cv::putText(frame, "No board detected",
                cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);

        cv::imshow("Calibration", frame);
        char key = cv::waitKey(30);

        if (key == 's' && last_found) {
            corner_list.push_back(corner_set);
            point_list.push_back(buildPointSet());
            std::cout << "Frame saved! Total: " << corner_list.size() << std::endl;
        }

        if (key == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}