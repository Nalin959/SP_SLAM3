#include "SuperPoint.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include "cuda/SuperPoint.h"
#include "cuda_utils.h"

namespace ORB_SLAM3
{

SuperPoint::SuperPoint(const std::string& engine_path) : max_h(1080), max_w(1920), mUseCLAHE(false) {
    trt_model = std::make_shared<TRTModel>(engine_path);
    cudaStreamCreate(&stream);
    
    // Allocate GPU buffers for Batch Size 2
    // Image input: 2 x 1 x max_h x max_w (float32)
    cudaMalloc(&buffers[0], 2 * 1 * max_h * max_w * sizeof(float));
    // Prob output: 2 x 65 x (max_h/8) x (max_w/8) (float32)
    cudaMalloc(&buffers[1], 2 * 65 * (max_h/8) * (max_w/8) * sizeof(float));
    // Desc output: 2 x 256 x (max_h/8) * (max_w/8) (float32)
    cudaMalloc(&buffers[2], 2 * 256 * (max_h/8) * (max_w/8) * sizeof(float));
    // prob_out: 2 x max_h x max_w (float32)
    cudaMalloc(&buffers[3], 2 * max_h * max_w * sizeof(float));

    if (!trt_model->getContext()) {
        std::cerr << "Failed to create context!" << std::endl;
    }

    // Read CLAHE preference directly from environment variable
    const char* env_p = std::getenv("SP_USE_CLAHE");
    if (env_p) {
        mUseCLAHE = (std::string(env_p) == "1");
    } else {
        // Default to ON for backward compatibility with EO tracking
        mUseCLAHE = true;
    }
}

SuperPoint::~SuperPoint() {
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);
    cudaFree(buffers[3]);
    cudaStreamDestroy(stream);
}

void SuperPoint::forward(const cv::Mat& image) {
    int H = image.rows;
    int W = image.cols;
    last_H = H;
    last_W = W;
    if (H > max_h || W > max_w) {
        std::cerr << "[SuperPoint] Error: Image size " << W << "x" << H << " exceeds allocated " << max_w << "x" << max_h << std::endl;
        return;
    }

    // Set dynamic input shape
    auto context = trt_model->getContext();
    if (!context) return;
    
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = 1;
    input_dims.d[1] = 1;
    input_dims.d[2] = H;
    input_dims.d[3] = W;
    context->setInputShape("image", input_dims);

    cv::Mat image_f;
    if (mUseCLAHE) {
        cv::Mat processed;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(image, processed);
        processed.convertTo(image_f, CV_32F, 1.0/255.0);
    } else {
        image.convertTo(image_f, CV_32F, 1.0/255.0);
    }

    // Copy to GPU
    cudaMemcpyAsync(buffers[0], image_f.data, H * W * sizeof(float), cudaMemcpyHostToDevice, stream);

    context->setTensorAddress("image", buffers[0]);
    context->setTensorAddress("prob", buffers[1]);
    context->setTensorAddress("desc", buffers[2]);

    // Run inference
    trt_model->enqueue(stream);
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) std::cerr << "CUDA Error after enqueue: " << cudaGetErrorString(err) << std::endl;

    // Call CUDA Softmax & Fold
    run_superpoint_softmax_fold_cuda((float*)buffers[1], (float*)buffers[3], H, W, stream);
    cudaStreamSynchronize(stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) std::cerr << "CUDA Error after fold1: " << cudaGetErrorString(err) << std::endl;
}

void SuperPoint::forwardCUDA_Batch(float* d_image_f_b1, float* d_image_f_b2, int H, int W) {
    float* d_batch_buffer = (float*)buffers[0];
    cudaMemcpyAsync(d_batch_buffer, d_image_f_b1, H * W * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_batch_buffer + H * W, d_image_f_b2, H * W * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    last_H = H;
    last_W = W;
    if (H > max_h || W > max_w) {
        std::cerr << "[SuperPoint] Error: Image size " << W << "x" << H << " exceeds allocated " << max_w << "x" << max_h << std::endl;
        return;
    }

    auto context = trt_model->getContext();
    if (!context) return;
    
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = 2; // Fixed batch size 2 for pipeline
    input_dims.d[1] = 1;
    input_dims.d[2] = H;
    input_dims.d[3] = W;
    context->setInputShape("image", input_dims);

    context->setTensorAddress("image", d_batch_buffer);
    context->setTensorAddress("prob", buffers[1]);
    context->setTensorAddress("desc", buffers[2]);

    trt_model->enqueue(stream);
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) std::cerr << "CUDA Error after enqueue: " << cudaGetErrorString(err) << std::endl;

    run_superpoint_softmax_fold_cuda((float*)buffers[1], (float*)buffers[3], H, W, stream);
    cudaStreamSynchronize(stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) std::cerr << "CUDA Error after fold1: " << cudaGetErrorString(err) << std::endl;
    
    int prob_size = 65 * (H/8) * (W/8);
    int out_size = H * W;
    run_superpoint_softmax_fold_cuda(((float*)buffers[1]) + prob_size, ((float*)buffers[3]) + out_size, H, W, stream);
    cudaStreamSynchronize(stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) std::cerr << "CUDA Error after fold2: " << cudaGetErrorString(err) << std::endl;
}

void SuperPoint::forwardCUDA(float* d_image_f, int H, int W) {
    last_H = H;
    last_W = W;
    if (H > max_h || W > max_w) {
        std::cerr << "[SuperPoint] Error: Image size " << W << "x" << H << " exceeds allocated " << max_w << "x" << max_h << std::endl;
        return;
    }

    auto context = trt_model->getContext();
    if (!context) return;
    
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = 1;
    input_dims.d[1] = 1;
    input_dims.d[2] = H;
    input_dims.d[3] = W;
    context->setInputShape("image", input_dims);

    context->setTensorAddress("image", d_image_f);
    context->setTensorAddress("prob", buffers[1]);
    context->setTensorAddress("desc", buffers[2]);

    trt_model->enqueue(stream);
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) std::cerr << "CUDA Error after enqueue: " << cudaGetErrorString(err) << std::endl;

    run_superpoint_softmax_fold_cuda((float*)buffers[1], (float*)buffers[3], H, W, stream);
    cudaStreamSynchronize(stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) std::cerr << "CUDA Error after fold1: " << cudaGetErrorString(err) << std::endl;
}

void SuperPoint::getKeyPointsCUDA_Batch(float threshold, int nms_radius, std::vector<cv::KeyPoint>& keypoints_b1, std::vector<cv::KeyPoint>& keypoints_b2, int max_kpts)
{
    if (max_kpts <= 0) return;
    
    cv::KeyPoint* d_kpts1 = nullptr;
    cv::KeyPoint* d_kpts2 = nullptr;
    int* d_kpt_count = nullptr;
    cudaMalloc(&d_kpts1, max_kpts * sizeof(cv::KeyPoint));
    cudaMalloc(&d_kpts2, max_kpts * sizeof(cv::KeyPoint));
    cudaMalloc(&d_kpt_count, 2 * sizeof(int));
    cudaMemset(d_kpt_count, 0, 2 * sizeof(int));
    
    cuda_superpoint_nms(
        (const float*)buffers[3], last_W, last_H, threshold, nms_radius,
        d_kpts1, &d_kpt_count[0], max_kpts, stream
    );
    
    cuda_superpoint_nms(
        ((const float*)buffers[3]) + last_W * last_H, last_W, last_H, threshold, nms_radius,
        d_kpts2, &d_kpt_count[1], max_kpts, stream
    );

    int h_count[2] = {0, 0};
    cudaMemcpyAsync(h_count, d_kpt_count, 2 * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (h_count[0] > max_kpts) h_count[0] = max_kpts;
    if (h_count[1] > max_kpts) h_count[1] = max_kpts;
    
    if (h_count[0] > 0) {
        cv::KeyPoint* h_kpts = new cv::KeyPoint[h_count[0]];
        cudaMemcpyAsync(h_kpts, d_kpts1, h_count[0] * sizeof(cv::KeyPoint), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        keypoints_b1.assign(h_kpts, h_kpts + h_count[0]);
        delete[] h_kpts;
    }
    
    if (h_count[1] > 0) {
        cv::KeyPoint* h_kpts = new cv::KeyPoint[h_count[1]];
        cudaMemcpyAsync(h_kpts, d_kpts2, h_count[1] * sizeof(cv::KeyPoint), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        keypoints_b2.assign(h_kpts, h_kpts + h_count[1]);
        delete[] h_kpts;
    }

    cudaFree(d_kpts1);
    cudaFree(d_kpts2);
    cudaFree(d_kpt_count);
}

void SuperPoint::getKeyPointsCUDA(float threshold, int nms_radius, std::vector<cv::KeyPoint>& keypoints, int max_kpts)
{
    if (max_kpts <= 0) return;
    
    cv::KeyPoint* d_kpts = nullptr;
    int* d_kpt_count = nullptr;
    cudaMalloc(&d_kpts, max_kpts * sizeof(cv::KeyPoint));
    cudaMalloc(&d_kpt_count, sizeof(int));
    cudaMemset(d_kpt_count, 0, sizeof(int));
    
    cuda_superpoint_nms(
        (const float*)buffers[3], last_W, last_H, threshold, nms_radius,
        d_kpts, d_kpt_count, max_kpts, stream
    );

    int h_count = 0;
    cudaMemcpyAsync(&h_count, d_kpt_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (h_count > max_kpts) h_count = max_kpts;
    
    if (h_count > 0) {
        keypoints.resize(h_count);
        cudaMemcpyAsync(keypoints.data(), d_kpts, h_count * sizeof(cv::KeyPoint), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    } else {
        keypoints.clear();
    }
    
    cudaFree(d_kpts);
    cudaFree(d_kpt_count);
}

void SuperPoint::computeDescriptorsCUDA_Batch(const std::vector<cv::KeyPoint>& kpts1, const std::vector<cv::KeyPoint>& kpts2, cv::Mat& desc1, cv::Mat& desc2) {
    computeDescriptorsCUDA(kpts1, desc1);
    
    // For desc2, we need to temporarily shift the desc buffer pointer!
    void* orig_desc = buffers[2];
    int desc_size = 256 * (last_H/8) * (last_W/8);
    buffers[2] = ((float*)orig_desc) + desc_size;
    
    computeDescriptorsCUDA(kpts2, desc2);
    
    buffers[2] = orig_desc;
}

void SuperPoint::computeDescriptorsCUDA(const std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    if (keypoints.empty()) return;
    
    // Allocate space for CPU descriptors
    descriptors.create(keypoints.size(), 256, CV_32F);

    // Call CUDA descriptor grid_sample and L2 normalize
    run_superpoint_desc_cuda(
        (float*)buffers[2], 
        keypoints, 
        (float*)descriptors.data, 
        last_H, last_W, 
        stream
    );
    
    cudaStreamSynchronize(stream);
}

SPDetector::SPDetector(std::shared_ptr<SuperPoint> _model, bool use_fp16)
    : model(_model), mbFP16(use_fp16)
{
}

void SPDetector::detect(cv::Mat &image, bool cuda)
{
    model->forward(image);
}

void SPDetector::detectCUDA_Batch(float* d_image_f_b1, float* d_image_f_b2, int H, int W) {
    model->forwardCUDA_Batch(d_image_f_b1, d_image_f_b2, H, W);
}

void SPDetector::detectCUDA(float* d_image_f, int H, int W)
{
    model->forwardCUDA(d_image_f, H, W);
}

void SPDetector::getKeyPoints_Batch(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &kpts1, std::vector<cv::KeyPoint> &kpts2, bool nms) {
    model->getKeyPointsCUDA_Batch(threshold, 4, kpts1, kpts2, 4096);
}

void SPDetector::getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms)
{
    model->getKeyPointsCUDA(threshold, nms ? 4 : 0, keypoints, 100000);
    
    // Filter by bounding box
    if (iniX > 0 || iniY > 0 || maxX < 10000 || maxY < 10000) {
        std::vector<cv::KeyPoint> filtered;
        filtered.reserve(keypoints.size());
        for (const auto& kp : keypoints) {
            if (kp.pt.x >= iniX && kp.pt.x < maxX && kp.pt.y >= iniY && kp.pt.y < maxY) {
                cv::KeyPoint shifted = kp;
                shifted.pt.x -= iniX;
                shifted.pt.y -= iniY;
                filtered.push_back(shifted);
            }
        }
        keypoints = std::move(filtered);
    }
}

void SPDetector::computeDescriptors_Batch(const std::vector<cv::KeyPoint> &kpts1, const std::vector<cv::KeyPoint> &kpts2, cv::Mat &desc1, cv::Mat &desc2) {
    model->computeDescriptorsCUDA_Batch(kpts1, kpts2, desc1, desc2);
}

void SPDetector::computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    model->computeDescriptorsCUDA(keypoints, descriptors);
}

} // namespace ORB_SLAM3
