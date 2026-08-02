#ifndef LIGHTGLUE_H
#define LIGHTGLUE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <mutex>
#include <memory>
#include "TRTModel.h"

namespace ORB_SLAM3
{

struct LightGlueMatch {
    int idx0;       // keypoint index in image 0
    int idx1;       // keypoint index in image 1
    float score;    // match confidence
};

class LightGlue {
public:
    LightGlue(const std::string &engine_path);
    ~LightGlue();

    // Match two sets of keypoints + descriptors
    // keypoints: [N,2] (x,y) pixel coordinates
    // descriptors: [N,256] float descriptors
    // image_size: (width, height) for coordinate normalization
    std::vector<LightGlueMatch> match(
        const std::vector<cv::KeyPoint> &kpts0,
        const cv::Mat &desc0,
        const std::vector<cv::KeyPoint> &kpts1,
        const cv::Mat &desc1,
        const cv::Size &image_size);

    bool isLoaded() const { return mbLoaded; }

    // Thread-safe GPU inference mutex (shared across all LightGlue instances)
    static std::mutex& getInferenceMutex() {
        static std::mutex mtx;
        return mtx;
    }

private:
    std::shared_ptr<TRTModel> trt_model;
    bool mbLoaded;
    cudaStream_t stream;
    void* buffers[6]; // kpts0, kpts1, desc0, desc1, matches0, scores0
    int max_kpts;
};

}  // namespace ORB_SLAM3

#endif // LIGHTGLUE_H
