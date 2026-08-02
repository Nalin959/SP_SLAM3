#ifndef SUPERPOINT_H
#define SUPERPOINT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "TRTModel.h"

namespace ORB_SLAM3
{

class SuperPoint {
public:
    SuperPoint(const std::string& engine_path);
    ~SuperPoint();

    void forward(const cv::Mat& image, cv::Mat& prob);
    void computeDescriptorsCUDA(const std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

private:
    std::shared_ptr<TRTModel> trt_model;
    void* buffers[4]; // 0: image, 1: prob, 2: desc, 3: prob_out_hw // 0: input, 1: prob, 2: desc
    cudaStream_t stream;
    int max_h, max_w;
    int last_H, last_W;
    bool mUseCLAHE;
};

class SPDetector {
public:
    SPDetector(std::shared_ptr<SuperPoint> _model, bool use_fp16 = false);
    void detect(cv::Mat &image, bool cuda);
    void getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms);
    void computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

private:
    std::shared_ptr<SuperPoint> model;
    cv::Mat mProb; // H x W probability map (CPU)
    bool mbFP16;
};

}  // namespace ORB_SLAM3

#endif
