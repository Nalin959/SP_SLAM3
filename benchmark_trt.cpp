#include "TRTModel.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <vector>

using namespace ORB_SLAM3;

cv::Mat preprocessImage(const cv::Mat &image) {
    cv::Mat img;
    if (image.channels() == 1)
        cv::cvtColor(image, img, cv::COLOR_GRAY2RGB);
    else if (image.channels() == 3)
        img = image;
    else
        cv::cvtColor(image, img, cv::COLOR_BGRA2RGB);

    cv::resize(img, img, cv::Size(320, 320));
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);

    cv::Mat channels[3];
    cv::split(img, channels);
    channels[0] = (channels[0] - 0.485f) / 0.229f;
    channels[1] = (channels[1] - 0.456f) / 0.224f;
    channels[2] = (channels[2] - 0.406f) / 0.225f;
    
    cv::Mat blob;
    cv::merge(channels, 3, img);
    blob.create(1, 3 * 320 * 320, CV_32F);
    float* blob_data = blob.ptr<float>();
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < 320 * 320; ++i) {
            blob_data[c * 320 * 320 + i] = channels[c].at<float>(i);
        }
    }
    return blob;
}

int main() {
    std::cout << "Loading TensorRT Engines..." << std::endl;
    TRTModel sp_model("superpoint.engine");
    TRTModel lg_model("lightglue.engine");
    TRTModel cp_model("cosplace.engine");

    if (!sp_model.getEngine() || !lg_model.getEngine() || !cp_model.getEngine()) {
        std::cerr << "Failed to load engines." << std::endl;
        return 1;
    }

    // Load images
    cv::Mat img1 = cv::imread("test_frame.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("reference_map.png", cv::IMREAD_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Could not read images." << std::endl;
        return 1;
    }
    
    cv::resize(img1, img1, cv::Size(640, 480));
    cv::resize(img2, img2, cv::Size(640, 480));

    // Buffers for SuperPoint
    void* sp_buffers[3];
    cudaMalloc(&sp_buffers[0], 1 * 1 * img1.rows * img1.cols * sizeof(float)); // Input
    cudaMalloc(&sp_buffers[1], 1 * 65 * (img1.rows / 8) * (img1.cols / 8) * sizeof(float)); // Scores
    cudaMalloc(&sp_buffers[2], 1 * 256 * (img1.rows / 8) * (img1.cols / 8) * sizeof(float)); // Descriptors
    
    sp_model.getContext()->setInputShape("image", nvinfer1::Dims4{1, 1, img1.rows, img1.cols});
    sp_model.getContext()->setTensorAddress("image", sp_buffers[0]);
    sp_model.getContext()->setTensorAddress("prob", sp_buffers[1]);
    sp_model.getContext()->setTensorAddress("desc", sp_buffers[2]);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup SuperPoint
    sp_model.enqueue(stream);
    cudaStreamSynchronize(stream);

    std::cout << "Benchmarking SuperPoint (10 iterations)..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        sp_model.enqueue(stream);
    }
    cudaStreamSynchronize(stream);
    auto t2 = std::chrono::high_resolution_clock::now();
    double sp_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0 / 10.0;
    std::cout << "SuperPoint Extraction (1 image): " << sp_time << " ms" << std::endl;

    // Buffers for LightGlue
    void* lg_buffers[6];
    cudaMalloc(&lg_buffers[0], 1 * 1024 * 2 * sizeof(float)); // kpts0
    cudaMalloc(&lg_buffers[1], 1 * 1024 * 2 * sizeof(float)); // kpts1
    cudaMalloc(&lg_buffers[2], 1 * 1024 * 256 * sizeof(float)); // desc0
    cudaMalloc(&lg_buffers[3], 1 * 1024 * 256 * sizeof(float)); // desc1
    cudaMalloc(&lg_buffers[4], 1024 * 2 * sizeof(int64_t)); // matches0
    cudaMalloc(&lg_buffers[5], 1024 * sizeof(float)); // scores0

    lg_model.getContext()->setInputShape("kpts0", nvinfer1::Dims3{1, 1024, 2});
    lg_model.getContext()->setInputShape("kpts1", nvinfer1::Dims3{1, 1024, 2});
    lg_model.getContext()->setInputShape("desc0", nvinfer1::Dims3{1, 1024, 256});
    lg_model.getContext()->setInputShape("desc1", nvinfer1::Dims3{1, 1024, 256});

    lg_model.getContext()->setTensorAddress("kpts0", lg_buffers[0]);
    lg_model.getContext()->setTensorAddress("kpts1", lg_buffers[1]);
    lg_model.getContext()->setTensorAddress("desc0", lg_buffers[2]);
    lg_model.getContext()->setTensorAddress("desc1", lg_buffers[3]);
    lg_model.getContext()->setTensorAddress("matches0", lg_buffers[4]);
    lg_model.getContext()->setTensorAddress("scores0", lg_buffers[5]);

    // Warmup LightGlue
    lg_model.enqueue(stream);
    cudaStreamSynchronize(stream);

    std::cout << "Benchmarking LightGlue (10 iterations)..." << std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        lg_model.enqueue(stream);
    }
    cudaStreamSynchronize(stream);
    t2 = std::chrono::high_resolution_clock::now();
    double lg_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0 / 10.0;
    std::cout << "LightGlue Matching (1024 keypoints): " << lg_time << " ms" << std::endl;

    // Buffers for CosPlace
    void* cp_buffers[2];
    cudaMalloc(&cp_buffers[0], 1 * 3 * 320 * 320 * sizeof(float)); // Input
    cudaMalloc(&cp_buffers[1], 1 * 512 * sizeof(float)); // Output
    
    cp_model.getContext()->setTensorAddress("image", cp_buffers[0]);
    cp_model.getContext()->setTensorAddress("descriptor", cp_buffers[1]);

    // Warmup CosPlace
    cp_model.enqueue(stream);
    cudaStreamSynchronize(stream);

    std::cout << "Benchmarking CosPlace (10 iterations)..." << std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        cp_model.enqueue(stream);
    }
    cudaStreamSynchronize(stream);
    t2 = std::chrono::high_resolution_clock::now();
    double cp_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0 / 10.0;
    std::cout << "CosPlace Extraction (1 image): " << cp_time << " ms" << std::endl;

    // Cleanup
    cudaFree(sp_buffers[0]); cudaFree(sp_buffers[1]); cudaFree(sp_buffers[2]);
    cudaFree(lg_buffers[0]); cudaFree(lg_buffers[1]); cudaFree(lg_buffers[2]); cudaFree(lg_buffers[3]);
    cudaFree(cp_buffers[0]); cudaFree(cp_buffers[1]);
    cudaStreamDestroy(stream);

    return 0;
}
