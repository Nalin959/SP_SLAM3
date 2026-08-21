#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <opencv2/core.hpp>

void cuda_bgr2gray_resize_normalize(const uint8_t* d_in, int in_w, int in_h, int in_step, float* d_out, int out_w, int out_h, cudaStream_t stream);
void cuda_crop_map_normalize(const uint8_t* d_in, int in_w, int in_h, int in_step, float* d_out, int out_w, int out_h, float center_x, float center_y, float angle_rad, cudaStream_t stream);
void cuda_superpoint_nms(const float* d_prob, int w, int h, float threshold, int nms_radius, cv::KeyPoint* d_kpts, int* d_kpt_count, int max_kpts, cudaStream_t stream);
void cuda_render_visualization(float* d_frame_f, float* d_map_f, uchar3* d_out, float2* d_pts1, float2* d_pts2, int num_matches, cudaStream_t stream);
