#include "TRTModel.h"
#include <iostream>
#include <fstream>
#include <vector>

namespace ORB_SLAM3
{

TRTModel::TRTModel(const std::string& enginePath) : mRuntime(nullptr), mEngine(nullptr), mContext(nullptr) {
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        std::cerr << "Failed to open TRT engine file: " << enginePath << std::endl;
        return;
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Failed to read TRT engine file." << std::endl;
        return;
    }
    
    mRuntime = nvinfer1::createInferRuntime(mLogger);
    if (!mRuntime) {
        std::cerr << "Failed to create TRT runtime." << std::endl;
        return;
    }
    
    mEngine = mRuntime->deserializeCudaEngine(buffer.data(), size);
    if (!mEngine) {
        std::cerr << "Failed to deserialize TRT engine." << std::endl;
        return;
    }
    
    mContext = mEngine->createExecutionContext();
    if (!mContext) {
        std::cerr << "Failed to create execution context." << std::endl;
    }
}

TRTModel::~TRTModel() {
    if (mContext) delete mContext;
    if (mEngine) delete mEngine;
    if (mRuntime) delete mRuntime;
}

bool TRTModel::enqueue(cudaStream_t stream) {
    if (!mContext) return false;
    
    // Note: TensorRT 8.5+ uses enqueueV3
    // Assuming context bindings are set by the caller (e.g., setTensorAddress)
    return mContext->enqueueV3(stream);
}

} // namespace ORB_SLAM3
