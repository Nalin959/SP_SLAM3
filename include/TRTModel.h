#ifndef TRTMODEL_H
#define TRTMODEL_H

#include <NvInfer.h>
#include <string>
#include <cuda_runtime_api.h>
#include <iostream>

namespace ORB_SLAM3
{

class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

class TRTModel {
public:
    TRTModel(const std::string& enginePath);
    ~TRTModel();

    bool enqueue(cudaStream_t stream);

    nvinfer1::IExecutionContext* getContext() const { return mContext; }
    nvinfer1::ICudaEngine* getEngine() const { return mEngine; }

private:
    TRTLogger mLogger;
    nvinfer1::IRuntime* mRuntime;
    nvinfer1::ICudaEngine* mEngine;
    nvinfer1::IExecutionContext* mContext;
};

} // namespace ORB_SLAM3

#endif // TRTMODEL_H
