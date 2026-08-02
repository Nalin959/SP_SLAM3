#include "cuda/PlaceRecognition.h"
#include <iostream>

namespace ORB_SLAM3
{

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
        } \
    } while (0)

__global__ void preprocess_imagenet_kernel(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int in_H, int in_W, int in_step, int in_channels,
    int out_H, int out_W,
    bool is_bgr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_W || y >= out_H) return;

    // Bilinear interpolation coordinates
    float scale_x = (float)in_W / out_W;
    float scale_y = (float)in_H / out_H;

    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;

    int x0 = floorf(src_x);
    int y0 = floorf(src_y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float dx = src_x - x0;
    float dy = src_y - y0;

    auto read_pixel = [&](int py, int px, int c) -> float {
        px = max(0, min(px, in_W - 1));
        py = max(0, min(py, in_H - 1));
        if (in_channels == 1) {
            return (float)input[py * in_step + px];
        } else {
            return (float)input[py * in_step + px * in_channels + c];
        }
    };

    float out_c[3];
    for (int c = 0; c < 3; ++c) {
        float p00 = read_pixel(y0, x0, c);
        float p10 = read_pixel(y0, x1, c);
        float p01 = read_pixel(y1, x0, c);
        float p11 = read_pixel(y1, x1, c);

        out_c[c] = (p00 * (1.0f - dx) + p10 * dx) * (1.0f - dy) +
                   (p01 * (1.0f - dx) + p11 * dx) * dy;
    }

    // Convert BGR to RGB if necessary
    float r = is_bgr ? out_c[2] : out_c[0];
    float g = out_c[1];
    float b = is_bgr ? out_c[0] : out_c[2];

    // ImageNet normalization
    r = (r / 255.0f - 0.485f) / 0.229f;
    g = (g / 255.0f - 0.456f) / 0.224f;
    b = (b / 255.0f - 0.406f) / 0.225f;

    // Write to CHW format
    int area = out_H * out_W;
    int idx = y * out_W + x;
    output[0 * area + idx] = r;
    output[1 * area + idx] = g;
    output[2 * area + idx] = b;
}

void run_placerecognition_preprocess_cuda(
    const unsigned char* d_input_image, 
    int in_H, int in_W, int in_step, int in_channels,
    float* d_output_blob, 
    int out_H, int out_W,
    bool is_bgr,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((out_W + block.x - 1) / block.x, (out_H + block.y - 1) / block.y);

    preprocess_imagenet_kernel<<<grid, block, 0, stream>>>(
        d_input_image, d_output_blob, 
        in_H, in_W, in_step, in_channels,
        out_H, out_W, 
        is_bgr);
}

} // namespace ORB_SLAM3
