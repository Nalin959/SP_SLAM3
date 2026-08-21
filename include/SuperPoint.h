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

    void forward(const cv::Mat& image);
    void forwardCUDA(float* d_image_f, int H, int W);
    void forwardCUDA_Batch(float* d_image_f_b1, float* d_image_f_b2, int H, int W);
    void getKeyPointsCUDA(float threshold, int nms_radius, std::vector<cv::KeyPoint>& keypoints, int max_kpts);
    void getKeyPointsCUDA_Batch(float threshold, int nms_radius, std::vector<cv::KeyPoint>& keypoints_b1, std::vector<cv::KeyPoint>& keypoints_b2, int max_kpts);
    void computeDescriptorsCUDA(const std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    void computeDescriptorsCUDA_Batch(const std::vector<cv::KeyPoint>& kpts1, const std::vector<cv::KeyPoint>& kpts2, cv::Mat& desc1, cv::Mat& desc2);

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
    void detectCUDA(float* d_image_f, int H, int W);
    void detectCUDA_Batch(float* d_image_f_b1, float* d_image_f_b2, int H, int W);
    void getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms);
    void getKeyPoints_Batch(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &kpts1, std::vector<cv::KeyPoint> &kpts2, bool nms);
    void computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);
    void computeDescriptors_Batch(const std::vector<cv::KeyPoint> &kpts1, const std::vector<cv::KeyPoint> &kpts2, cv::Mat &desc1, cv::Mat &desc2);

private:
    std::shared_ptr<SuperPoint> model;

    bool mbFP16;
};

}  // namespace ORB_SLAM3

#endif
