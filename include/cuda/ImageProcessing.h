#ifndef IMAGEPROCESSING_CU_H
#define IMAGEPROCESSING_CU_H

#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>

void rgb2gray_custom_cuda(cv::Mat& img, bool is_rgb);
void resize_custom_cuda(const cv::Mat& src, cv::Mat& dst, cv::Size sz);
void copy_make_border_custom_cuda(const cv::Mat& src, cv::Mat& dst, int top, int bottom, int left, int right);

#endif // IMAGEPROCESSING_CU_H
