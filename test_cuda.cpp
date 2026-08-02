#include "include/SuperPoint.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace ORB_SLAM3;

int main() {
    cv::Mat img = cv::imread("./test_frame.png", cv::IMREAD_GRAYSCALE);
    if(img.empty()) { cout << "No image" << endl; return 1; }
    cv::resize(img, img, cv::Size(640, 480));
    
    SuperPoint sp("superpoint.engine");
    cv::Mat prob;
    sp.forward(img, prob);
    
    // Extract keypoints like SPDetector
    std::vector<cv::KeyPoint> kpts;
    for(int y=0; y<prob.rows; y++) {
        for(int x=0; x<prob.cols; x++) {
            if(prob.at<float>(y, x) > 0.015f) {
                kpts.push_back(cv::KeyPoint(x, y, 8.0f));
            }
        }
    }
    cout << "Found " << kpts.size() << " keypoints" << endl;
    
    if(kpts.empty()) return 0;
    
    cv::Mat desc;
    sp.computeDescriptorsCUDA(kpts, desc);
    
    cout << "Desc shape: " << desc.rows << "x" << desc.cols << endl;
    cout << "Desc[0]: " << desc.at<float>(0, 0) << ", " << desc.at<float>(0, 1) << endl;
    
    // Let's also test L2 norm of the first descriptor
    float norm = 0;
    for(int i=0; i<256; ++i) norm += desc.at<float>(0, i) * desc.at<float>(0, i);
    cout << "Desc[0] norm: " << sqrt(norm) << endl;
    
    return 0;
}
