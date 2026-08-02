#include "SuperPoint.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include "cuda/SuperPoint.h"

namespace ORB_SLAM3
{

SuperPoint::SuperPoint(const std::string& engine_path) : max_h(768), max_w(1024) {
    trt_model = std::make_shared<TRTModel>(engine_path);
    cudaStreamCreate(&stream);
    
    // Allocate GPU buffers
    // Image input: 1 x 1 x max_h x max_w (float32)
    cudaMalloc(&buffers[0], 1 * 1 * max_h * max_w * sizeof(float));
    // Prob output: 1 x 65 x (max_h/8) x (max_w/8) (float32)
    cudaMalloc(&buffers[1], 65 * (max_h/8) * (max_w/8) * sizeof(float));
    cudaMalloc(&buffers[2], 1 * 256 * (max_h/8) * (max_w/8) * sizeof(float));
    // prob_out: max_h x max_w (float32)
    cudaMalloc(&buffers[3], max_h * max_w * sizeof(float));

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

void SuperPoint::forward(const cv::Mat& image, cv::Mat& prob) {
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

    // Call CUDA Softmax & Fold
    run_superpoint_softmax_fold_cuda((float*)buffers[1], (float*)buffers[3], H, W, stream);

    // Prepare CPU output
    prob.create(H, W, CV_32F);

    // Copy back folded HxW probability map to CPU
    cudaMemcpyAsync(prob.data, buffers[3], H * W * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
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
    model->forward(image, mProb);
}

void SPDetector::getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms)
{
    // PyTorch SuperPoint discards 4 pixels from borders
    int pad = 4;
    // Simple NMS (3x3) + threshold
    for (int y = iniY + pad; y < maxY - pad; ++y) {
        for (int x = iniX + pad; x < maxX - pad; ++x) {
            float val = mProb.at<float>(y, x);
            if (val > threshold) {
                bool is_max = true;
                if (nms) {
                    for (int y_offset = -4; y_offset <= 4; ++y_offset) {
                        for (int x_offset = -4; x_offset <= 4; ++x_offset) {
                            if (y_offset == 0 && x_offset == 0) continue;
                            int ny = y + y_offset;
                            int nx = x + x_offset;
                            if (ny >= 0 && ny < mProb.rows && nx >= 0 && nx < mProb.cols) {
                                if (mProb.at<float>(ny, nx) > val) {
                                    is_max = false;
                                    break;
                                }
                            }
                        }
                        if (!is_max) break;
                    }
                }
                if (is_max) {
                    keypoints.emplace_back(cv::Point2f(x - iniX, y - iniY), 8.0f, -1.0f, val);
                }
            }
        }
    }
}

void SPDetector::computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    model->computeDescriptorsCUDA(keypoints, descriptors);
}

} // namespace ORB_SLAM3
