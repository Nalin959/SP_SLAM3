#ifndef PLACE_RECOGNITION_H
#define PLACE_RECOGNITION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <mutex>
#include <set>
#include <memory>
#include "TRTModel.h"

namespace ORB_SLAM3
{

class KeyFrame;

class PlaceRecognition {
public:
    PlaceRecognition(const std::string &engine_path);
    ~PlaceRecognition();

    // Extract global descriptor from an image
    cv::Mat extractDescriptor(const cv::Mat &image);

    // Add a keyframe to the database
    void add(KeyFrame *pKF, const cv::Mat &descriptor);

    // Query N most similar keyframes, excluding connected ones
    std::vector<KeyFrame*> query(KeyFrame *pKF,
                                  const cv::Mat &descriptor,
                                  int nCandidates,
                                  const std::set<KeyFrame*> &spConnectedKFs);

    // Remove a keyframe from the database
    void erase(KeyFrame *pKF);

    bool isLoaded() const { return mbLoaded; }

private:
    std::shared_ptr<TRTModel> trt_model;
    bool mbLoaded;
    cudaStream_t stream;
    void* buffers[3]; // input, output, raw image

    // Database: stores descriptors for cosine similarity search
    struct Entry {
        KeyFrame* pKF;
        cv::Mat descriptor;  // [1 x 512] float32 normalized
    };
    std::vector<Entry> mvDatabase;
    std::mutex mMutex;

    cv::Mat preprocessImage(const cv::Mat &image);
};

}  // namespace ORB_SLAM3

#endif // PLACE_RECOGNITION_H
