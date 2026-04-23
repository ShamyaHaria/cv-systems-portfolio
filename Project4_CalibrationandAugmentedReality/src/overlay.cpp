// Shamya Haria
// Project 4: Extension - Texture Overlay on Checkerboard

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

const int BOARD_COLS = 9;
const int BOARD_ROWS = 6;

std::vector<cv::Vec3f> buildPointSet() {
    std::vector<cv::Vec3f> point_set;
    for (int r = 0; r < BOARD_ROWS; r++)
        for (int c = 0; c < BOARD_COLS; c++)
            point_set.push_back(cv::Vec3f(c, -r, 0));
    return point_set;
}

void drawAxes(cv::Mat &frame, cv::Mat &camera_matrix, cv::Mat &dist_coeffs,
              cv::Mat &rvec, cv::Mat &tvec) {
    std::vector<cv::Vec3f> axisPoints = {
        {0,0,0}, {3,0,0}, {0,-3,0}, {0,0,3}
    };
    std::vector<cv::Point2f> ip;
    cv::projectPoints(axisPoints, rvec, tvec, camera_matrix, dist_coeffs, ip);

    cv::line(frame, ip[0], ip[1], cv::Scalar(0,0,255), 3);
    cv::line(frame, ip[0], ip[2], cv::Scalar(0,255,0), 3);
    cv::line(frame, ip[0], ip[3], cv::Scalar(255,0,0), 3);
    cv::putText(frame, "X", ip[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
    cv::putText(frame, "Y", ip[2], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
    cv::putText(frame, "Z", ip[3], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,0,0), 2);
}

void drawVirtualObject(cv::Mat &frame, cv::Mat &camera_matrix, cv::Mat &dist_coeffs,
                       cv::Mat &rvec, cv::Mat &tvec) {
    std::vector<cv::Vec3f> pts = {
        {2,-1,-2}, {6,-1,-2}, {6,-4,-2}, {2,-4,-2},
        {2,-1,-4}, {6,-1,-4}, {6,-4,-4}, {2,-4,-4},
        {1,-2.5f,-5.5f}, {7,-2.5f,-5.5f},
        {5,-2,-5.5f}, {5.8f,-2,-5.5f},
        {5,-2,-4}, {5.8f,-2,-4}
    };
    std::vector<cv::Point2f> ip;
    cv::projectPoints(pts, rvec, tvec, camera_matrix, dist_coeffs, ip);

    cv::Scalar yellow(0,255,255);
    cv::Scalar cyan(255,255,0);
    cv::Scalar magenta(255,0,255);

    cv::line(frame, ip[0], ip[1], yellow, 2);
    cv::line(frame, ip[1], ip[2], yellow, 2);
    cv::line(frame, ip[2], ip[3], yellow, 2);
    cv::line(frame, ip[3], ip[0], yellow, 2);
    cv::line(frame, ip[4], ip[5], yellow, 2);
    cv::line(frame, ip[5], ip[6], yellow, 2);
    cv::line(frame, ip[6], ip[7], yellow, 2);
    cv::line(frame, ip[7], ip[4], yellow, 2);
    cv::line(frame, ip[0], ip[4], yellow, 2);
    cv::line(frame, ip[1], ip[5], yellow, 2);
    cv::line(frame, ip[2], ip[6], yellow, 2);
    cv::line(frame, ip[3], ip[7], yellow, 2);
    cv::line(frame, ip[4], ip[8], cyan, 2);
    cv::line(frame, ip[5], ip[9], cyan, 2);
    cv::line(frame, ip[7], ip[8], cyan, 2);
    cv::line(frame, ip[6], ip[9], cyan, 2);
    cv::line(frame, ip[8], ip[9], cyan, 3);
    cv::line(frame, ip[10], ip[11], magenta, 2);
    cv::line(frame, ip[12], ip[13], magenta, 2);
    cv::line(frame, ip[10], ip[12], magenta, 2);
    cv::line(frame, ip[11], ip[13], magenta, 2);
}

int main() {
    cv::Mat camera_matrix, dist_coeffs;
    cv::FileStorage fs("calibration.yml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Cannot open calibration.yml" << std::endl;
        return -1;
    }
    fs["camera_matrix"] >> camera_matrix;
    fs["dist_coeffs"] >> dist_coeffs;
    fs.release();

    cv::Mat overlay = cv::imread("data/chessboard.png");
    if (overlay.empty()) {
        std::cerr << "Cannot load data/chessboard.png" << std::endl;
        return -1;
    }

    std::cout << "Calibration and overlay image loaded." << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  't' - toggle texture overlay" << std::endl;
    std::cout << "  'a' - toggle axes" << std::endl;
    std::cout << "  'o' - toggle virtual object" << std::endl;
    std::cout << "  'q' - quit" << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        return -1;
    }

    std::vector<cv::Vec3f> point_set = buildPointSet();
    bool showTexture = true;
    bool showAxes = true;
    bool showObject = true;

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

            if (showTexture) {
                // Project 4 outer corners of board
                std::vector<cv::Vec3f> boardCorners = {
                    {0, 0, 0}, {8, 0, 0}, {8, -5, 0}, {0, -5, 0}
                };
                std::vector<cv::Point2f> projCorners;
                cv::projectPoints(boardCorners, rvec, tvec, camera_matrix, dist_coeffs, projCorners);

                // Source corners of overlay image
                std::vector<cv::Point2f> srcCorners = {
                    {0, 0},
                    {(float)overlay.cols, 0},
                    {(float)overlay.cols, (float)overlay.rows},
                    {0, (float)overlay.rows}
                };

                // Compute homography and warp overlay onto board
                cv::Mat H = cv::findHomography(srcCorners, projCorners);
                cv::Mat warped;
                cv::warpPerspective(overlay, warped, H, frame.size());

                // Create mask from warped image
                cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
                std::vector<cv::Point> poly;
                for (auto &p : projCorners)
                    poly.push_back(cv::Point(p.x, p.y));
                cv::fillConvexPoly(mask, poly, cv::Scalar(255));

                // Blend warped overlay onto frame
                warped.copyTo(frame, mask);
            }

            if (showAxes) drawAxes(frame, camera_matrix, dist_coeffs, rvec, tvec);
            if (showObject) drawVirtualObject(frame, camera_matrix, dist_coeffs, rvec, tvec);
        }

        cv::putText(frame, found ? "Board detected" : "No board",
            cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
            found ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), 2);

        cv::imshow("Overlay AR", frame);
        char key = cv::waitKey(30);

        if (key == 't') showTexture = !showTexture;
        if (key == 'a') showAxes = !showAxes;
        if (key == 'o') showObject = !showObject;
        if (key == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}