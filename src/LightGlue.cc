#include "LightGlue.h"
#include <iostream>
#include <cuda_runtime_api.h>

namespace ORB_SLAM3
{

LightGlue::LightGlue(const std::string &engine_path) 
    : mbLoaded(false), max_kpts(1024)
{
    trt_model = std::make_shared<TRTModel>(engine_path);
    if (!trt_model->getEngine()) {
        std::cerr << "Failed to load LightGlue TensorRT engine: " << engine_path << std::endl;
        return;
    }
    
    cudaStreamCreate(&stream);
    
    // Allocate GPU buffers for max_kpts
    // kpts0: 1 x max_kpts x 2 (float32)
    cudaMalloc(&buffers[0], 1 * max_kpts * 2 * sizeof(float));
    // kpts1: 1 x max_kpts x 2 (float32)
    cudaMalloc(&buffers[1], 1 * max_kpts * 2 * sizeof(float));
    // desc0: 1 x max_kpts x 256 (float32)
    cudaMalloc(&buffers[2], 1 * max_kpts * 256 * sizeof(float));
    // desc1: 1 x max_kpts x 256 (float32)
    cudaMalloc(&buffers[3], 1 * max_kpts * 256 * sizeof(float));
    // matches0: max_kpts x 2 (int32_t)
    cudaMalloc(&buffers[4], max_kpts * 2 * sizeof(int32_t));
    // scores0: max_kpts (float32)
    cudaMalloc(&buffers[5], max_kpts * sizeof(float));

    if (trt_model->getContext()) {
        mbLoaded = true;
    }
}

LightGlue::~LightGlue() {
    for (int i = 0; i < 6; ++i) {
        if (buffers[i]) cudaFree(buffers[i]);
    }
    cudaStreamDestroy(stream);
}

std::vector<LightGlueMatch> LightGlue::match(
    const std::vector<cv::KeyPoint> &kpts0,
    const cv::Mat &desc0,
    const std::vector<cv::KeyPoint> &kpts1,
    const cv::Mat &desc1,
    const cv::Size &image_size)
{
    std::vector<LightGlueMatch> matches;

    if (!mbLoaded || kpts0.empty() || kpts1.empty())
        return matches;

    int N = std::min((int)kpts0.size(), max_kpts);
    int M = std::min((int)kpts1.size(), max_kpts);

    std::lock_guard<std::mutex> lock(getInferenceMutex());

    auto context = trt_model->getContext();
    if (!context) return matches;

    // Set dynamic shapes
    nvinfer1::Dims dims_kpts0{3, {1, N, 2}};
    nvinfer1::Dims dims_kpts1{3, {1, M, 2}};
    nvinfer1::Dims dims_desc0{3, {1, N, 256}};
    nvinfer1::Dims dims_desc1{3, {1, M, 256}};
    
    context->setInputShape("kpts0", dims_kpts0);
    context->setInputShape("kpts1", dims_kpts1);
    context->setInputShape("desc0", dims_desc0);
    context->setInputShape("desc1", dims_desc1);

    // Prepare host buffers
    std::vector<float> h_kpts0(N * 2);
    std::vector<float> h_kpts1(M * 2);
    
    float w = (float)image_size.width;
    float h = (float)image_size.height;
    
    // PyTorch LightGlue normalizes by max(width, height) to preserve aspect ratio:
    // shift = size / 2, scale = max(size) / 2
    // kpts = (kpts - shift) / scale
    float scale = std::max(w, h) / 2.0f;
    float shift_x = w / 2.0f;
    float shift_y = h / 2.0f;

    for (int i = 0; i < N; ++i) {
        h_kpts0[2*i] = (kpts0[i].pt.x - shift_x) / scale;
        h_kpts0[2*i+1] = (kpts0[i].pt.y - shift_y) / scale;
    }
    for (int i = 0; i < M; ++i) {
        h_kpts1[2*i] = (kpts1[i].pt.x - shift_x) / scale;
        h_kpts1[2*i+1] = (kpts1[i].pt.y - shift_y) / scale;
    }

    // Copy inputs to GPU
    cudaMemcpyAsync(buffers[0], h_kpts0.data(), N * 2 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[1], h_kpts1.data(), M * 2 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[2], desc0.data, N * 256 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[3], desc1.data, M * 256 * sizeof(float), cudaMemcpyHostToDevice, stream);

    context->setTensorAddress("kpts0", buffers[0]);
    context->setTensorAddress("kpts1", buffers[1]);
    context->setTensorAddress("desc0", buffers[2]);
    context->setTensorAddress("desc1", buffers[3]);
    context->setTensorAddress("matches0", buffers[4]);
    context->setTensorAddress("scores0", buffers[5]);

    // Run inference
    trt_model->enqueue(stream);

    // Get output shapes
    nvinfer1::Dims match_dims = context->getTensorShape("matches0");
    nvinfer1::Dims score_dims = context->getTensorShape("scores0");

    // Determine actual number of match pairs
    int K_matches = 0;
    bool is_pair_format = false;
    
    if (match_dims.nbDims == 2 && match_dims.d[1] == 2) {
        K_matches = match_dims.d[0];
        is_pair_format = true;
    } else if (match_dims.nbDims == 1) {
        int flat_size = match_dims.d[0];
        int score_size = (score_dims.nbDims >= 1) ? score_dims.d[0] : 0;
        if (score_size > 0 && score_size == flat_size / 2) {
            K_matches = score_size;
            is_pair_format = true;
        } else {
            K_matches = flat_size;
            is_pair_format = false;
        }
    }

    if (K_matches > 0 && K_matches <= max_kpts) {
        if (is_pair_format) {
            int total_elements = K_matches * 2;
            std::vector<int32_t> h_matches(total_elements);
            std::vector<float> h_scores(K_matches);
            cudaMemcpyAsync(h_matches.data(), buffers[4], total_elements * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_scores.data(), buffers[5], K_matches * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            matches.reserve(K_matches);
            for (int i = 0; i < K_matches; ++i) {
                int idx0 = (int)h_matches[2 * i];
                int idx1 = (int)h_matches[2 * i + 1];
                if (idx0 >= 0 && idx0 < N && idx1 >= 0 && idx1 < M) {
                    LightGlueMatch m;
                    m.idx0 = idx0;
                    m.idx1 = idx1;
                    m.score = h_scores[i];
                    matches.push_back(m);
                }
            }
        } else {
            std::vector<int64_t> h_matches(K_matches);
            std::vector<float> h_scores(K_matches);
            cudaMemcpyAsync(h_matches.data(), buffers[4], K_matches * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_scores.data(), buffers[5], K_matches * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            matches.reserve(K_matches);
            for (int i = 0; i < K_matches; ++i) {
                int idx1 = (int)h_matches[i];
                if (idx1 >= 0 && idx1 < M) {
                    LightGlueMatch m;
                    m.idx0 = i;
                    m.idx1 = idx1;
                    m.score = h_scores[i];
                    matches.push_back(m);
                }
            }
        }
    } else {
        cudaStreamSynchronize(stream);
    }

    return matches;
}

} // namespace ORB_SLAM3
