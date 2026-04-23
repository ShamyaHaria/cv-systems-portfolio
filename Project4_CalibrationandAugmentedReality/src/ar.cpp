// Shamya Haria
// Project 4: Calibration and Augmented Reality
// Task 4: solvePnP, Task 5: Project axes, Task 6: Virtual object

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

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

// Task 5: Draw 3D axes on the board
void drawAxes(cv::Mat &frame, cv::Mat &camera_matrix, cv::Mat &dist_coeffs,
              cv::Mat &rvec, cv::Mat &tvec) {
    std::vector<cv::Vec3f> axisPoints = {
        {0, 0, 0},  // origin
        {3, 0, 0},  // X axis - red
        {0, -3, 0}, // Y axis - green
        {0, 0, 3}   // Z axis - blue (towards viewer)
    };

    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(axisPoints, rvec, tvec, camera_matrix, dist_coeffs, imagePoints);

    cv::line(frame, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 3);   // X - red
    cv::line(frame, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 3);   // Y - green
    cv::line(frame, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 3);   // Z - blue

    cv::putText(frame, "X", imagePoints[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
    cv::putText(frame, "Y", imagePoints[2], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
    cv::putText(frame, "Z", imagePoints[3], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,0,0), 2);
}

// Task 6: Draw a virtual object - a house shape (more complex than cube)
void drawVirtualObject(cv::Mat &frame, cv::Mat &camera_matrix, cv::Mat &dist_coeffs,
                       cv::Mat &rvec, cv::Mat &tvec) {

    // House base (floor plan centered on board)
    // All Z values negative = floating above board
    std::vector<cv::Vec3f> pts = {
        // Base rectangle (z = -2, floating 2 units above board)
        {2, -1, -2}, {6, -1, -2}, {6, -4, -2}, {2, -4, -2},  // 0,1,2,3
        // Top rectangle (z = -4)
        {2, -1, -4}, {6, -1, -4}, {6, -4, -4}, {2, -4, -4},  // 4,5,6,7
        // Roof peak points
        {1, -2.5f, -5.5f}, {7, -2.5f, -5.5f},                 // 8,9 - roof ridge
        // Chimney
        {5, -2, -5.5f}, {5.8f, -2, -5.5f},                    // 10,11
        {5, -2, -4},    {5.8f, -2, -4}                         // 12,13
    };

    std::vector<cv::Point2f> ip;
    cv::projectPoints(pts, rvec, tvec, camera_matrix, dist_coeffs, ip);

    cv::Scalar yellow(0, 255, 255);
    cv::Scalar cyan(255, 255, 0);
    cv::Scalar magenta(255, 0, 255);

    // Base rectangle
    cv::line(frame, ip[0], ip[1], yellow, 2);
    cv::line(frame, ip[1], ip[2], yellow, 2);
    cv::line(frame, ip[2], ip[3], yellow, 2);
    cv::line(frame, ip[3], ip[0], yellow, 2);

    // Top rectangle
    cv::line(frame, ip[4], ip[5], yellow, 2);
    cv::line(frame, ip[5], ip[6], yellow, 2);
    cv::line(frame, ip[6], ip[7], yellow, 2);
    cv::line(frame, ip[7], ip[4], yellow, 2);

    // Vertical walls
    cv::line(frame, ip[0], ip[4], yellow, 2);
    cv::line(frame, ip[1], ip[5], yellow, 2);
    cv::line(frame, ip[2], ip[6], yellow, 2);
    cv::line(frame, ip[3], ip[7], yellow, 2);

    // Roof
    cv::line(frame, ip[4], ip[8], cyan, 2);
    cv::line(frame, ip[5], ip[9], cyan, 2);
    cv::line(frame, ip[7], ip[8], cyan, 2);
    cv::line(frame, ip[6], ip[9], cyan, 2);
    cv::line(frame, ip[8], ip[9], cyan, 3); // ridge line

    // Chimney
    cv::line(frame, ip[10], ip[11], magenta, 2);
    cv::line(frame, ip[12], ip[13], magenta, 2);
    cv::line(frame, ip[10], ip[12], magenta, 2);
    cv::line(frame, ip[11], ip[13], magenta, 2);
}

void drawOuterCorners(cv::Mat &frame, cv::Mat &camera_matrix, cv::Mat &dist_coeffs,
                      cv::Mat &rvec, cv::Mat &tvec) {
    std::vector<cv::Vec3f> corners = {
        {0,0,0}, {8,0,0}, {0,-5,0}, {8,-5,0}
    };
    std::vector<cv::Point2f> ip;
    cv::projectPoints(corners, rvec, tvec, camera_matrix, dist_coeffs, ip);

    cv::circle(frame, ip[0], 8, cv::Scalar(0,255,255), -1);
    cv::circle(frame, ip[1], 8, cv::Scalar(0,255,255), -1);
    cv::circle(frame, ip[2], 8, cv::Scalar(0,255,255), -1);
    cv::circle(frame, ip[3], 8, cv::Scalar(0,255,255), -1);
    cv::line(frame, ip[0], ip[1], cv::Scalar(0,255,255), 2);
    cv::line(frame, ip[1], ip[3], cv::Scalar(0,255,255), 2);
    cv::line(frame, ip[3], ip[2], cv::Scalar(0,255,255), 2);
    cv::line(frame, ip[2], ip[0], cv::Scalar(0,255,255), 2);
}

void drawRocket(cv::Mat &frame, cv::Mat &camera_matrix, cv::Mat &dist_coeffs,
                cv::Mat &rvec, cv::Mat &tvec) {
    std::vector<cv::Vec3f> pts = {
        {3.5f,-2,-2}, {4.5f,-2,-2}, {4.5f,-3,-2}, {3.5f,-3,-2},
        {3.5f,-2,-5}, {4.5f,-2,-5}, {4.5f,-3,-5}, {3.5f,-3,-5},
        {4.0f,-2.5f,-7},
        {3.5f,-2,-2}, {3.0f,-2,-1}, {3.5f,-2,-3},
        {4.5f,-2,-2}, {5.0f,-2,-1}, {4.5f,-2,-3},
        {3.5f,-3,-2}, {3.5f,-3.5f,-1}, {4.5f,-3,-2},
        {3.7f,-2.2f,-2}, {4.3f,-2.2f,-2},
        {3.5f,-2.5f,-1}, {4.5f,-2.5f,-1}
    };
    std::vector<cv::Point2f> ip;
    cv::projectPoints(pts, rvec, tvec, camera_matrix, dist_coeffs, ip);

    cv::Scalar white(255,255,255);
    cv::Scalar orange(0,165,255);
    cv::Scalar red(0,0,255);

    cv::line(frame, ip[0], ip[1], white, 2);
    cv::line(frame, ip[1], ip[2], white, 2);
    cv::line(frame, ip[2], ip[3], white, 2);
    cv::line(frame, ip[3], ip[0], white, 2);
    cv::line(frame, ip[4], ip[5], white, 2);
    cv::line(frame, ip[5], ip[6], white, 2);
    cv::line(frame, ip[6], ip[7], white, 2);
    cv::line(frame, ip[7], ip[4], white, 2);
    cv::line(frame, ip[0], ip[4], white, 2);
    cv::line(frame, ip[1], ip[5], white, 2);
    cv::line(frame, ip[2], ip[6], white, 2);
    cv::line(frame, ip[3], ip[7], white, 2);
    cv::line(frame, ip[4], ip[8], orange, 2);
    cv::line(frame, ip[5], ip[8], orange, 2);
    cv::line(frame, ip[6], ip[8], orange, 2);
    cv::line(frame, ip[7], ip[8], orange, 2);
    cv::line(frame, ip[9],  ip[10], orange, 2);
    cv::line(frame, ip[10], ip[11], orange, 2);
    cv::line(frame, ip[11], ip[9],  orange, 2);
    cv::line(frame, ip[12], ip[13], orange, 2);
    cv::line(frame, ip[13], ip[14], orange, 2);
    cv::line(frame, ip[14], ip[12], orange, 2);
    cv::line(frame, ip[15], ip[16], orange, 2);
    cv::line(frame, ip[16], ip[17], orange, 2);
    cv::line(frame, ip[17], ip[15], orange, 2);
    cv::line(frame, ip[18], ip[20], red, 2);
    cv::line(frame, ip[19], ip[21], red, 2);
    cv::line(frame, ip[18], ip[21], red, 2);
    cv::line(frame, ip[19], ip[20], red, 2);
}

int main(int argc, char* argv[]) {
    // Load calibration
    cv::Mat camera_matrix, dist_coeffs;
    cv::FileStorage fs("calibration.yml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Cannot open calibration.yml" << std::endl;
        return -1;
    }
    fs["camera_matrix"] >> camera_matrix;
    fs["dist_coeffs"] >> dist_coeffs;
    fs.release();

    std::cout << "Calibration loaded." << std::endl;

    std::vector<cv::Vec3f> point_set = buildPointSet();

    // Extension 1: Static image mode
    if (argc > 1) {
        std::cout << "Static image mode" << std::endl;
        for (int i = 1; i < argc; i++) {
            cv::Mat img = cv::imread(argv[i]);
            if (img.empty()) {
                std::cout << "Cannot load: " << argv[i] << std::endl;
                continue;
            }

            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

            std::vector<cv::Point2f> corner_set;
            bool found = cv::findChessboardCorners(
                gray, cv::Size(BOARD_COLS, BOARD_ROWS), corner_set,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
            );

            if (found) {
                cv::cornerSubPix(gray, corner_set, cv::Size(11,11), cv::Size(-1,-1),
                    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));

                cv::Mat rvec, tvec;
                cv::solvePnP(point_set, corner_set, camera_matrix, dist_coeffs, rvec, tvec);

                drawAxes(img, camera_matrix, dist_coeffs, rvec, tvec);
                drawVirtualObject(img, camera_matrix, dist_coeffs, rvec, tvec);

                std::string outname = "results/ar_static_" + std::to_string(i) + ".png";
                cv::imwrite(outname, img);
                std::cout << "Saved: " << outname << std::endl;

                cv::imshow("AR Static: " + std::string(argv[i]), img);
                cv::waitKey(0);
            } else {
                std::cout << "No board found in: " << argv[i] << std::endl;
            }
        }
        return 0;
    }

    // Live camera mode
    std::cout << "Controls:" << std::endl;
    std::cout << "  'a' - toggle axes" << std::endl;
    std::cout << "  'o' - toggle virtual object" << std::endl;
    std::cout << "  'r' - toggle rocket" << std::endl;
    std::cout << "  'c' - toggle corners" << std::endl;
    std::cout << "  'q' - quit" << std::endl;

    bool showAxes = true;
    bool showObject = true;
    bool showRocket = false;
    bool showCorners = true;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        return -1;
    }

    cv::Mat frame, gray;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corner_set;
        bool found = cv::findChessboardCorners(
            gray, cv::Size(BOARD_COLS, BOARD_ROWS), corner_set,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK
        );

        if (found) {
            cv::cornerSubPix(gray, corner_set, cv::Size(11,11), cv::Size(-1,-1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));

            cv::Mat rvec, tvec;
            cv::solvePnP(point_set, corner_set, camera_matrix, dist_coeffs, rvec, tvec);

            std::string rtext = "R: " + std::to_string(rvec.at<double>(0)).substr(0,5) + " " +
                             std::to_string(rvec.at<double>(1)).substr(0,5) + " " +
                             std::to_string(rvec.at<double>(2)).substr(0,5);
            std::string ttext = "T: " + std::to_string(tvec.at<double>(0)).substr(0,5) + " " +
                             std::to_string(tvec.at<double>(1)).substr(0,5) + " " +
                             std::to_string(tvec.at<double>(2)).substr(0,5);
            cv::putText(frame, rtext, cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,0), 2);
            cv::putText(frame, ttext, cv::Point(10,90), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,0), 2);

            if (showAxes) drawAxes(frame, camera_matrix, dist_coeffs, rvec, tvec);
            if (showCorners) drawOuterCorners(frame, camera_matrix, dist_coeffs, rvec, tvec);
            if (showObject) drawVirtualObject(frame, camera_matrix, dist_coeffs, rvec, tvec);
            if (showRocket) drawRocket(frame, camera_matrix, dist_coeffs, rvec, tvec);
        }

        cv::putText(frame, found ? "Board detected" : "No board",
            cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
            found ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), 2);

        cv::imshow("AR", frame);
        char key = cv::waitKey(30);

        if (key == 'a') showAxes = !showAxes;
        if (key == 'o') showObject = !showObject;
        if (key == 'r') showRocket = !showRocket;
        if (key == 'c') showCorners = !showCorners;
        if (key == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}