#include "cuda_utils.h"
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

__global__ void bgr2gray_resize_normalize_kernel(
    const uint8_t* __restrict__ bgr, int in_w, int in_h, int in_step,
    float* __restrict__ out, int out_w, int out_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_w && y < out_h) {
        float src_x = (x + 0.5f) * in_w / out_w - 0.5f;
        float src_y = (y + 0.5f) * in_h / out_h - 0.5f;
        
        int x0 = (int)floorf(src_x);
        int y0 = (int)floorf(src_y);
        int x1 = min(x0 + 1, in_w - 1);
        int y1 = min(y0 + 1, in_h - 1);
        x0 = max(x0, 0);
        y0 = max(y0, 0);

        float dx = src_x - x0;
        float dy = src_y - y0;

        auto get_gray = [&](int px, int py) {
            const uint8_t* p = bgr + py * in_step + px * 3;
            // BGR to Gray: 0.114*B + 0.587*G + 0.299*R
            return 0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2];
        };

        float v00 = get_gray(x0, y0);
        float v10 = get_gray(x1, y0);
        float v01 = get_gray(x0, y1);
        float v11 = get_gray(x1, y1);

        float val = (1 - dx) * (1 - dy) * v00 +
                    dx * (1 - dy) * v10 +
                    (1 - dx) * dy * v01 +
                    dx * dy * v11;

        out[y * out_w + x] = val / 255.0f;
    }
}

void cuda_bgr2gray_resize_normalize(
    const uint8_t* d_bgr, int in_w, int in_h, int in_step,
    float* d_out, int out_w, int out_h, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);
    bgr2gray_resize_normalize_kernel<<<grid, block, 0, stream>>>(d_bgr, in_w, in_h, in_step, d_out, out_w, out_h);
}

__global__ void crop_map_normalize_kernel(
    const uint8_t* __restrict__ map, int map_w, int map_h, int map_step,
    float* __restrict__ out, int out_w, int out_h,
    float cx, float cy, float cos_a, float sin_a)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_w && y < out_h) {
        int roi_x = (int)(cx - 700.0f);
        int roi_y = (int)(cy - 700.0f);
        float rot_cx = 699.5f;
        float rot_cy = 699.5f;
        float src_x = x + 188.0f;
        float src_y = y + 188.0f;
        float dx = src_x - rot_cx;
        float dy = src_y - rot_cy;
        float map_x = dx * cos_a + dy * sin_a + rot_cx + roi_x;
        float map_y = -dx * sin_a + dy * cos_a + rot_cy + roi_y;

        float val = 0.0f;
        if (map_x >= 0 && map_x < map_w - 1 && map_y >= 0 && map_y < map_h - 1) {
            int x0 = (int)floorf(map_x);
            int y0 = (int)floorf(map_y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            float fx = map_x - x0;
            float fy = map_y - y0;
            
            auto get_pixel = [&](int px, int py) {
                return (float)map[py * map_step + px];
            };

            float v00 = get_pixel(x0, y0);
            float v10 = get_pixel(x1, y0);
            float v01 = get_pixel(x0, y1);
            float v11 = get_pixel(x1, y1);

            val = (1 - fx) * (1 - fy) * v00 +
                  fx * (1 - fy) * v10 +
                  (1 - fx) * fy * v01 +
                  fx * fy * v11;
        }
        out[y * out_w + x] = val / 255.0f;
    }
}

void cuda_crop_map_normalize(
    const uint8_t* d_map, int map_w, int map_h, int map_step,
    float* d_out, int out_w, int out_h,
    float center_x, float center_y, float angle_rad,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);
    float cos_a = cosf(angle_rad);
    float sin_a = sinf(angle_rad);
    crop_map_normalize_kernel<<<grid, block, 0, stream>>>(d_map, map_w, map_h, map_step, d_out, out_w, out_h, center_x, center_y, cos_a, sin_a);
}

// Simple NMS kernel
__global__ void superpoint_nms_kernel(
    const float* __restrict__ prob, int w, int h, float threshold, int radius,
    cv::KeyPoint* kpts, int* kpt_count, int max_kpts)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= radius && x < w - radius && y >= radius && y < h - radius) {
        float val = prob[y * w + x];
        if (val > threshold) {
            bool is_max = true;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (prob[(y + dy) * w + (x + dx)] > val) {
                        is_max = false;
                        break;
                    }
                }
                if (!is_max) break;
            }
            if (is_max) {
                int idx = atomicAdd(kpt_count, 1);
                if (idx < max_kpts) {
                    kpts[idx].pt.x = x;
                    kpts[idx].pt.y = y;
                    kpts[idx].response = val;
                }
            }
        }
    }
}

void cuda_superpoint_nms(
    const float* d_prob, int w, int h, float threshold, int nms_radius,
    cv::KeyPoint* d_kpts, int* d_kpt_count, int max_kpts,
    cudaStream_t stream)
{
    cudaMemsetAsync(d_kpt_count, 0, sizeof(int), stream);
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    superpoint_nms_kernel<<<grid, block, 0, stream>>>(d_prob, w, h, threshold, nms_radius, d_kpts, d_kpt_count, max_kpts);
}
#include <cuda_runtime.h>
#include <iostream>

// Kernel to copy and convert float normalized (0..1) to uchar3 (BGR)
// grid is 2048x1024
__global__ void render_bg_kernel(float* d_frame_f, float* d_map_f, uchar3* d_out, int cols, int rows) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    
    int out_idx = y * cols + x;
    float val;
    if (x < 1024) {
        val = d_frame_f[y * 1024 + x];
    } else {
        val = d_map_f[y * 1024 + (x - 1024)];
    }
    
    unsigned char c = (unsigned char)(fminf(fmaxf(val * 255.0f, 0.0f), 255.0f));
    d_out[out_idx] = make_uchar3(c, c, c); // grayscale as BGR
}

__device__ void draw_circle(uchar3* d_out, int cols, int rows, int cx, int cy, int radius, uchar3 color) {
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            if (x*x + y*y <= radius*radius) {
                int px = cx + x;
                int py = cy + y;
                if (px >= 0 && px < cols && py >= 0 && py < rows) {
                    d_out[py * cols + px] = color;
                }
            }
        }
    }
}

__device__ void draw_line(uchar3* d_out, int cols, int rows, int x0, int y0, int x1, int y1, uchar3 color) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = (dx > dy ? dx : -dy) / 2;
    int e2;
    
    while (true) {
        if (x0 >= 0 && x0 < cols && y0 >= 0 && y0 < rows) {
            d_out[y0 * cols + x0] = color;
        }
        if (x0 == x1 && y0 == y1) break;
        e2 = err;
        if (e2 > -dx) { err -= dy; x0 += sx; }
        if (e2 < dy) { err += dx; y0 += sy; }
    }
}

__global__ void render_matches_kernel(uchar3* d_out, int cols, int rows, float2* pts1, float2* pts2, int num_matches) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_matches) return;
    
    int cx1 = (int)pts1[idx].x;
    int cy1 = (int)pts1[idx].y;
    int cx2 = (int)pts2[idx].x + 1024;
    int cy2 = (int)pts2[idx].y;
    
    uchar3 color = make_uchar3(0, 255, 0); // Green
    
    draw_line(d_out, cols, rows, cx1, cy1, cx2, cy2, color);
    draw_circle(d_out, cols, rows, cx1, cy1, 3, color);
    draw_circle(d_out, cols, rows, cx2, cy2, 3, color);
}

void cuda_render_visualization(float* d_frame_f, float* d_map_f, uchar3* d_out, float2* d_pts1, float2* d_pts2, int num_matches, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((2048 + block.x - 1) / block.x, (1024 + block.y - 1) / block.y);
    render_bg_kernel<<<grid, block, 0, stream>>>(d_frame_f, d_map_f, d_out, 2048, 1024);
    
    if (num_matches > 0) {
        int m_blocks = (num_matches + 255) / 256;
        render_matches_kernel<<<m_blocks, 256, 0, stream>>>(d_out, 2048, 1024, d_pts1, d_pts2, num_matches);
    }
}
