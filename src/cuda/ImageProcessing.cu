#include "cuda/ImageProcessing.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void rgb2gray_kernel(const unsigned char* d_rgb, unsigned char* d_gray, int width, int height, int channels, bool is_rgb) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        int out_idx = y * width + x;

        float r, g, b;
        if (is_rgb) {
            r = d_rgb[idx];
            g = d_rgb[idx + 1];
            b = d_rgb[idx + 2];
        } else {
            b = d_rgb[idx];
            g = d_rgb[idx + 1];
            r = d_rgb[idx + 2];
        }

        float gray = 0.299f * r + 0.587f * g + 0.114f * b;
        d_gray[out_idx] = (unsigned char)gray;
    }
}

extern "C" void run_rgb2gray_cuda(const unsigned char* d_rgb, unsigned char* d_gray, int width, int height, int channels, bool is_rgb, cudaStream_t stream) {
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    rgb2gray_kernel<<<gridSize, blockSize, 0, stream>>>(d_rgb, d_gray, width, height, channels, is_rgb);
}

__global__ void resize_bilinear_kernel(const unsigned char* d_src, unsigned char* d_dst, int src_w, int src_h, int dst_w, int dst_h) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x < dst_w && dst_y < dst_h) {
        float x_ratio = ((float)(src_w - 1)) / dst_w;
        float y_ratio = ((float)(src_h - 1)) / dst_h;

        int x_l = (int)(x_ratio * dst_x);
        int y_l = (int)(y_ratio * dst_y);
        int x_h = (x_l + 1 < src_w) ? x_l + 1 : x_l;
        int y_h = (y_l + 1 < src_h) ? y_l + 1 : y_l;

        float x_weight = (x_ratio * dst_x) - x_l;
        float y_weight = (y_ratio * dst_y) - y_l;

        float a = d_src[y_l * src_w + x_l];
        float b = d_src[y_l * src_w + x_h];
        float c = d_src[y_h * src_w + x_l];
        float d = d_src[y_h * src_w + x_h];

        float pixel = a * (1 - x_weight) * (1 - y_weight) + 
                      b * x_weight * (1 - y_weight) + 
                      c * y_weight * (1 - x_weight) + 
                      d * x_weight * y_weight;

        d_dst[dst_y * dst_w + dst_x] = (unsigned char)pixel;
    }
}

extern "C" void run_resize_bilinear_cuda(const unsigned char* d_src, unsigned char* d_dst, int src_w, int src_h, int dst_w, int dst_h, cudaStream_t stream) {
    dim3 blockSize(32, 32);
    dim3 gridSize((dst_w + blockSize.x - 1) / blockSize.x, (dst_h + blockSize.y - 1) / blockSize.y);
    resize_bilinear_kernel<<<gridSize, blockSize, 0, stream>>>(d_src, d_dst, src_w, src_h, dst_w, dst_h);
}

void rgb2gray_custom_cuda(cv::Mat& img, bool is_rgb) {
    if (img.channels() == 1) return;
    cv::Mat img_cont = img.isContinuous() ? img : img.clone();
    cv::Mat gray(img_cont.rows, img_cont.cols, CV_8UC1);
    unsigned char *d_src, *d_dst;
    cudaMalloc((void**)&d_src, img_cont.total() * img_cont.elemSize());
    cudaMalloc((void**)&d_dst, gray.total() * gray.elemSize());
    cudaMemcpy(d_src, img_cont.data, img_cont.total() * img_cont.elemSize(), cudaMemcpyHostToDevice);
    run_rgb2gray_cuda(d_src, d_dst, img_cont.cols, img_cont.rows, img_cont.channels(), is_rgb, 0);
    cudaMemcpy(gray.data, d_dst, gray.total() * gray.elemSize(), cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
    img = gray;
}

void resize_custom_cuda(const cv::Mat& src, cv::Mat& dst, cv::Size sz) {
    cv::Mat src_cont = src.isContinuous() ? src : src.clone();
    cv::Mat dst_cont(sz, src_cont.type());
    unsigned char *d_src, *d_dst;
    cudaMalloc((void**)&d_src, src_cont.total() * src_cont.elemSize());
    cudaMalloc((void**)&d_dst, dst_cont.total() * dst_cont.elemSize());
    cudaMemcpy(d_src, src_cont.data, src_cont.total() * src_cont.elemSize(), cudaMemcpyHostToDevice);
    run_resize_bilinear_cuda(d_src, d_dst, src_cont.cols, src_cont.rows, dst_cont.cols, dst_cont.rows, 0);
    cudaMemcpy(dst_cont.data, d_dst, dst_cont.total() * dst_cont.elemSize(), cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
    dst_cont.copyTo(dst);
}

__global__ void copy_make_border_reflect101_kernel(const unsigned char* d_src, unsigned char* d_dst, int src_w, int src_h, int dst_w, int dst_h, int top, int bottom, int left, int right) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x < dst_w && dst_y < dst_h) {
        int src_x = dst_x - left;
        int src_y = dst_y - top;

        if (src_x < 0) src_x = -src_x;
        else if (src_x >= src_w) src_x = 2 * src_w - src_x - 2;

        if (src_y < 0) src_y = -src_y;
        else if (src_y >= src_h) src_y = 2 * src_h - src_y - 2;

        d_dst[dst_y * dst_w + dst_x] = d_src[src_y * src_w + src_x];
    }
}

extern "C" void run_copy_make_border_cuda(const unsigned char* d_src, unsigned char* d_dst, int src_w, int src_h, int dst_w, int dst_h, int top, int bottom, int left, int right, cudaStream_t stream) {
    dim3 blockSize(32, 32);
    dim3 gridSize((dst_w + blockSize.x - 1) / blockSize.x, (dst_h + blockSize.y - 1) / blockSize.y);
    copy_make_border_reflect101_kernel<<<gridSize, blockSize, 0, stream>>>(d_src, d_dst, src_w, src_h, dst_w, dst_h, top, bottom, left, right);
}

void copy_make_border_custom_cuda(const cv::Mat& src, cv::Mat& dst, int top, int bottom, int left, int right) {
    cv::Mat src_cont = src.isContinuous() ? src : src.clone();
    cv::Mat dst_cont(src_cont.rows + top + bottom, src_cont.cols + left + right, src_cont.type());
    unsigned char *d_src, *d_dst;
    cudaMalloc((void**)&d_src, src_cont.total() * src_cont.elemSize());
    cudaMalloc((void**)&d_dst, dst_cont.total() * dst_cont.elemSize());
    cudaMemcpy(d_src, src_cont.data, src_cont.total() * src_cont.elemSize(), cudaMemcpyHostToDevice);
    run_copy_make_border_cuda(d_src, d_dst, src_cont.cols, src_cont.rows, dst_cont.cols, dst_cont.rows, top, bottom, left, right, 0);
    cudaMemcpy(dst_cont.data, d_dst, dst_cont.total() * dst_cont.elemSize(), cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
    dst_cont.copyTo(dst);
}

