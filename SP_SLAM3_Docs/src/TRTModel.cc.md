# Documentation: `TRTModel.cc`

## High-Level Overview
The `TRTModel.cc` file is a lightweight, low-level wrapper around the **NVIDIA TensorRT** C++ API. 
In `SP_SLAM3`, deep learning models like SuperPoint were originally trained in PyTorch and exported to ONNX. To achieve real-time latency on edge devices (like the NVIDIA Orin NX), these ONNX models are compiled into highly optimized, hardware-specific TensorRT `.engine` files. 
This class is responsible for loading the serialized binary engine file from the disk, deserializing it into GPU memory, creating an execution context, and providing a clean interface to trigger asynchronous inference.

**Primary Dependencies:**
- `TRTModel.h`
- `NvInfer.h` (The core NVIDIA TensorRT library).

---

## Block-by-Block Breakdown

### 1. Deserialization and Initialization

```cpp
TRTModel::TRTModel(const std::string& enginePath)
```
**Explanation:** 
- The constructor takes the absolute path to a pre-compiled TensorRT engine (e.g., `"superpoint.engine"`).
- **Binary Reading:** It opens the file using `std::ifstream` in binary mode. It seeks to the end (`file.end`) to determine the exact byte size of the engine, allocates a `std::vector<char>` buffer, seeks back to the beginning, and reads the entire engine into system RAM in a single block.
- **TensorRT Runtime (`mRuntime`):** It creates an `nvinfer1::IRuntime` object. This is the entry point to the TensorRT API. It requires a Logger object (`mLogger`, typically defined in the header to capture engine warnings/errors).
- **Engine Instantiation (`mEngine`):** It calls `deserializeCudaEngine`, passing the raw binary buffer. This translates the serialized bytes into a live GPU-optimized neural network graph (the `ICudaEngine`).
- **Execution Context (`mContext`):** It creates an `IExecutionContext`. While the engine defines the network's structure and weights, the *context* holds the dynamic state during inference (like intermediate activation buffers). If you wanted to run multi-threaded inference on the same engine, you would create multiple contexts.

### 2. Resource Management (Destructor)

```cpp
TRTModel::~TRTModel()
```
**Explanation:** 
- Proper teardown of GPU resources is critical in a long-running C++ SLAM system to prevent VRAM leaks.
- It explicitly deletes the `mContext`, the `mEngine`, and the `mRuntime` in reverse order of their creation.
- *(Note: In modern TensorRT versions (8.x+), NVIDIA recommends using smart pointers or the `destroy()` method instead of raw `delete`, but this raw pointer destruction remains valid for backward compatibility).*

### 3. Asynchronous Inference Execution

```cpp
bool TRTModel::enqueue(cudaStream_t stream)
```
**Explanation:** 
- This is the trigger function that tells the GPU to run the neural network.
- **Asynchronous Execution:** It uses `enqueueV3(stream)`. This does *not* block the CPU. It simply queues the computation on the specified CUDA stream and returns immediately.
- The calling function (e.g., inside `SPDetector::detect`) is responsible for binding the input and output tensor memory addresses to the context *before* calling `enqueue`, and for calling `cudaStreamSynchronize` *after* to wait for the results.
- It returns a boolean indicating whether the enqueue operation was successfully scheduled by the CUDA driver.
