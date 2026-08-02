#ifndef SUPERPOINT_CUDA_H
#define SUPERPOINT_CUDA_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

namespace ORB_SLAM3
{

struct GPUKeypoint {
    float x;
    float y;
    float score;
};

// 1. Softmax and Fold: Converts TRT output into a H x W float array.
// d_prob: 65xH8xW8 output from TRT
// d_prob_out: HxW output (must be pre-allocated on GPU)
void run_superpoint_softmax_fold_cuda(
    const float* d_prob,
    float* d_prob_out,
    int H, int W,
    cudaStream_t stream
);

// 2. Descriptor grid_sample: Interpolates 256-D descriptors for given keypoints.
// d_desc: 256xH8xW8 output from TRT
// h_keypoints: Keypoints from CPU
// d_desc_out: kpt_count x 256 output (must be pre-allocated on GPU)
void run_superpoint_desc_cuda(
    const float* d_desc,
    const std::vector<cv::KeyPoint>& h_keypoints,
    float* d_desc_out,
    int H, int W,
    cudaStream_t stream
);

} // namespace ORB_SLAM3

#endif
