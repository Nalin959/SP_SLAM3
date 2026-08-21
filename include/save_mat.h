#pragma once
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
inline void save_device_float_mat(const float* d_mat, int w, int h, const char* filename) {
    cv::Mat cpu_f(h, w, CV_32F);
    cudaMemcpy(cpu_f.data, d_mat, w * h * sizeof(float), cudaMemcpyDeviceToHost);
    cv::Mat cpu_8u;
    cpu_f.convertTo(cpu_8u, CV_8U, 255.0);
    cv::imwrite(filename, cpu_8u);
}
