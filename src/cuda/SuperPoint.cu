#include "cuda/SuperPoint.h"
#include <iostream>
#include <cmath>

namespace ORB_SLAM3
{

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
        } \
    } while (0)

__global__ void superpoint_softmax_fold_kernel(
    const float* __restrict__ prob_in, // 65 x H8 x W8
    float* __restrict__ prob_out,      // H x W
    int H8, int W8)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W8 || y >= H8) return;

    int spatial_idx = y * W8 + x;
    int stride = H8 * W8;

    // Find max for numerical stability
    float max_val = -1e9f;
    for (int c = 0; c < 65; ++c) {
        float val = prob_in[c * stride + spatial_idx];
        if (val > max_val) max_val = val;
    }

    // Softmax
    float sum_exp = 0.0f;
    float exp_vals[65];
    for (int c = 0; c < 65; ++c) {
        float val = expf(prob_in[c * stride + spatial_idx] - max_val);
        exp_vals[c] = val;
        sum_exp += val;
    }

    // Fold into H x W (drop 65th channel)
    for (int c = 0; c < 64; ++c) {
        float prob = exp_vals[c] / sum_exp;
        int dy = c / 8;
        int dx = c % 8;
        int out_y = y * 8 + dy;
        int out_x = x * 8 + dx;
        prob_out[out_y * (W8 * 8) + out_x] = prob;
    }
}

__global__ void superpoint_desc_sample_kernel(
    const GPUKeypoint* __restrict__ kpts,
    int kpt_count,
    const float* __restrict__ desc_in, // 256 x H8 x W8
    float* __restrict__ desc_out,      // kpt_count x 256
    int H8, int W8)
{
    int kpt_idx = blockIdx.x;
    if (kpt_idx >= kpt_count) return;

    int c = threadIdx.x; // 0 to 255
    if (c >= 256) return;

    float kpt_x = kpts[kpt_idx].x;
    float kpt_y = kpts[kpt_idx].y;

    // PyTorch grid_sample align_corners=True for SuperPoint
    // Python: grid = kpts / (W/2) - 1; grid_sample(align_corners=True)
    // px_grid = (grid + 1)/2 * (W8 - 1) = (kpt_x / W) * (W8 - 1)
    float px = (kpt_x / (W8 * 8.0f)) * (W8 - 1.0f);
    float py = (kpt_y / (H8 * 8.0f)) * (H8 - 1.0f);

    int x0 = floorf(px);
    int y0 = floorf(py);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float dx = px - x0;
    float dy = py - y0;

    int stride = H8 * W8;
    const float* channel_data = desc_in + c * stride;

    auto get_val = [&](int x, int y) {
        if (x >= 0 && x < W8 && y >= 0 && y < H8) {
            return channel_data[y * W8 + x];
        }
        return 0.0f;
    };

    float v00 = get_val(x0, y0);
    float v10 = get_val(x1, y0);
    float v01 = get_val(x0, y1);
    float v11 = get_val(x1, y1);

    float val = (v00 * (1.0f - dx) + v10 * dx) * (1.0f - dy) +
                (v01 * (1.0f - dx) + v11 * dx) * dy;

    // Write to unnormalized output
    desc_out[kpt_idx * 256 + c] = val;

    // L2 Normalize using block reduction
    __shared__ float s_sum[256];
    s_sum[c] = val * val;
    __syncthreads();

    // Reduction
    for (int offset = 128; offset > 0; offset /= 2) {
        if (c < offset) {
            s_sum[c] += s_sum[c + offset];
        }
        __syncthreads();
    }

    float norm = sqrtf(s_sum[0]);
    if (norm > 1e-6f) {
        desc_out[kpt_idx * 256 + c] = val / norm;
    } else {
        desc_out[kpt_idx * 256 + c] = 0.0f;
    }
}

void run_superpoint_softmax_fold_cuda(
    const float* d_prob,
    float* d_prob_out,
    int H, int W,
    cudaStream_t stream)
{
    int H8 = H / 8;
    int W8 = W / 8;

    dim3 block(16, 16);
    dim3 grid((W8 + block.x - 1) / block.x, (H8 + block.y - 1) / block.y);
    superpoint_softmax_fold_kernel<<<grid, block, 0, stream>>>(d_prob, d_prob_out, H8, W8);
}

void run_superpoint_desc_cuda(
    const float* d_desc,
    const std::vector<cv::KeyPoint>& h_keypoints,
    float* d_desc_out,
    int H, int W,
    cudaStream_t stream)
{
    int kpt_count = h_keypoints.size();
    if (kpt_count == 0) return;

    int H8 = H / 8;
    int W8 = W / 8;

    // Allocate and copy keypoints to GPU
    std::vector<GPUKeypoint> h_kpts(kpt_count);
    for (int i = 0; i < kpt_count; ++i) {
        h_kpts[i].x = h_keypoints[i].pt.x;
        h_kpts[i].y = h_keypoints[i].pt.y;
        h_kpts[i].score = h_keypoints[i].response;
    }

    GPUKeypoint* d_kpts = nullptr;
    float* d_desc_out_gpu = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_kpts, kpt_count * sizeof(GPUKeypoint), stream));
    CUDA_CHECK(cudaMallocAsync(&d_desc_out_gpu, kpt_count * 256 * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_kpts, h_kpts.data(), kpt_count * sizeof(GPUKeypoint), cudaMemcpyHostToDevice, stream));

    // 256 threads per block, one block per keypoint
    dim3 block(256);
    dim3 grid(kpt_count);
    superpoint_desc_sample_kernel<<<grid, block, 0, stream>>>(d_kpts, kpt_count, d_desc, d_desc_out_gpu, H8, W8);

    // Copy descriptors back to CPU
    CUDA_CHECK(cudaMemcpyAsync(d_desc_out, d_desc_out_gpu, kpt_count * 256 * sizeof(float), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaFreeAsync(d_kpts, stream));
    CUDA_CHECK(cudaFreeAsync(d_desc_out_gpu, stream));
}

} // namespace ORB_SLAM3
