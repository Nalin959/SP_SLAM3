#ifndef PLACERECOGNITION_CUDA_H
#define PLACERECOGNITION_CUDA_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

namespace ORB_SLAM3
{

// Preprocess image for CosPlace
void run_placerecognition_preprocess_cuda(
    const unsigned char* d_input_image, 
    int in_H, int in_W, int in_step, int in_channels,
    float* d_output_blob, 
    int out_H, int out_W,
    bool is_bgr,
    cudaStream_t stream
);

} // namespace ORB_SLAM3

#endif
