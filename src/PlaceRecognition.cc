#include "PlaceRecognition.h"
#include "LightGlue.h"
#include "KeyFrame.h"
#include <iostream>
#include <algorithm>
#include <cuda_runtime_api.h>
#include "cuda/PlaceRecognition.h"

namespace ORB_SLAM3
{

PlaceRecognition::PlaceRecognition(const std::string &engine_path)
    : mbLoaded(false)
{
    trt_model = std::make_shared<TRTModel>(engine_path);
    if (!trt_model->getEngine()) {
        std::cerr << "Failed to load PlaceRecognition TensorRT engine: " << engine_path << std::endl;
        return;
    }
    
    cudaStreamCreate(&stream);
    
    // Allocate GPU buffers
    // input: 1 x 3 x 320 x 320
    cudaMalloc(&buffers[0], 1 * 3 * 320 * 320 * sizeof(float));
    // output: 1 x 512
    cudaMalloc(&buffers[1], 1 * 512 * sizeof(float));
    // raw image buffer: up to 1920x1080x4
    cudaMalloc(&buffers[2], 1920 * 1080 * 4 * sizeof(unsigned char));

    if (trt_model->getContext()) {
        trt_model->getContext()->setTensorAddress("image", buffers[0]);
        trt_model->getContext()->setTensorAddress("descriptor", buffers[1]);
        mbLoaded = true;
    }
}

PlaceRecognition::~PlaceRecognition() {
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);
    cudaStreamDestroy(stream);
}

cv::Mat PlaceRecognition::extractDescriptor(const cv::Mat &image)
{
    if (!mbLoaded || image.empty())
        return cv::Mat();

    std::lock_guard<std::mutex> lock(LightGlue::getInferenceMutex());

    // Check size limit
    if (image.rows > 1080 || image.cols > 1920) {
        std::cerr << "[PlaceRecognition] Image too large for preallocated buffer!" << std::endl;
        return cv::Mat();
    }

    // Determine input format
    int in_channels = image.channels();
    bool is_bgr = false;
    if (in_channels == 3) {
        // By default ORB-SLAM3 System converts to RGB if mSettings.rgb() is true,
        // but it's safe to assume it's RGB or BGR.
        // Usually, OpenCV loads BGR. We'll treat it as BGR to RGB conversion.
        // Wait, if it's already RGB, we shouldn't swap. But standard cv::Mat is BGR.
        // Let's assume it's BGR unless we know it's already RGB.
        // For CosPlace, standard is RGB. If it's already RGB, is_bgr should be false.
        // Let's just say is_bgr = true if it came from imread, but from tracking it might be converted.
        // Actually, the original code did:
        // if (image.channels() == 3) img = image; ... which means it didn't swap channels!
        // So we will pass is_bgr = false to match the old behavior.
        is_bgr = false; 
    }

    // Ensure continuous buffer if needed, though cudaMemcpy2D handles steps, 
    // we can just clone if not continuous for safety
    cv::Mat cont_image = image;
    if (!image.isContinuous()) {
        cont_image = image.clone();
    }

    // Copy raw image to GPU
    cudaMemcpyAsync(buffers[2], cont_image.data, cont_image.total() * cont_image.elemSize(), cudaMemcpyHostToDevice, stream);

    // Run preprocessing kernel
    run_placerecognition_preprocess_cuda(
        (unsigned char*)buffers[2], 
        cont_image.rows, cont_image.cols, cont_image.step, in_channels,
        (float*)buffers[0], 
        320, 320, 
        is_bgr, 
        stream
    );

    // trt_model enqueue uses buffers[0] as input and buffers[1] as output
    trt_model->enqueue(stream);
    
    cv::Mat output(1, 512, CV_32F);
    cudaMemcpyAsync(output.data, buffers[1], 512 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);

    // L2 normalize
    float norm = 0;
    for (int i = 0; i < 512; ++i) {
        float v = output.at<float>(0, i);
        norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (int i = 0; i < 512; ++i) {
            output.at<float>(0, i) /= norm;
        }
    }

    return output;
}

void PlaceRecognition::add(KeyFrame *pKF, const cv::Mat &descriptor)
{
    if (descriptor.empty()) return;

    std::unique_lock<std::mutex> lock(mMutex);
    mvDatabase.push_back({pKF, descriptor});
}

void PlaceRecognition::erase(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutex);
    mvDatabase.erase(
        std::remove_if(mvDatabase.begin(), mvDatabase.end(),
            [pKF](const Entry &e) { return e.pKF == pKF; }),
        mvDatabase.end());
}

std::vector<KeyFrame*> PlaceRecognition::query(
    KeyFrame *pKF,
    const cv::Mat &descriptor,
    int nCandidates,
    const std::set<KeyFrame*> &spConnectedKFs)
{
    std::vector<KeyFrame*> result;
    if (descriptor.empty()) return result;

    std::unique_lock<std::mutex> lock(mMutex);

    if (mvDatabase.empty()) return result;

    mvDatabase.erase(
        std::remove_if(mvDatabase.begin(), mvDatabase.end(),
            [](const Entry &e) { return e.pKF->isBad(); }),
        mvDatabase.end());

    std::vector<std::pair<float, KeyFrame*>> scores;
    scores.reserve(mvDatabase.size());

    for (auto &entry : mvDatabase) {
        if (entry.pKF == pKF) continue;
        if (spConnectedKFs.count(entry.pKF)) continue;
        if (entry.descriptor.cols != 512) continue;

        float similarity = (float)entry.descriptor.dot(descriptor);
        scores.push_back({similarity, entry.pKF});
    }

    if (scores.empty()) return result;

    std::sort(scores.begin(), scores.end(), [](const auto &a, const auto &b) {
        return a.first > b.first;
    });

    int k = std::min(nCandidates, (int)scores.size());
    result.reserve(k);
    for (int i = 0; i < k; i++) {
        result.push_back(scores[i].second);
    }

    return result;
}

} // namespace ORB_SLAM3
