# Documentation: `LightGlue.cc`

## High-Level Overview
The `LightGlue.cc` file implements the `LightGlue` class, which serves as a highly optimized, hardware-accelerated wrapper for the LightGlue deep neural network. 
Traditionally, ORB-SLAM3 relies on standard feature matching (e.g., comparing binary ORB descriptors using Hamming distance). However, in extremely challenging environments (e.g., drastic illumination changes, feature-poor corridors, or cross-modal camera setups like RGB-to-Thermal), traditional matching fails. 
This file replaces the standard matcher with a **Transformer-based Neural Matcher (LightGlue)**, executed at blistering speeds using **NVIDIA TensorRT** and **LibTorch (PyTorch C++ API)**.

**Primary Dependencies:**
- `LightGlue.h`
- LibTorch (`torch::Tensor`) for CPU/GPU memory management.
- TensorRT (`TRTModel`) for compiled, optimized neural network execution.
- CUDA (`cudaStream_t`) for asynchronous GPU operations.

---

## Block-by-Block Breakdown

### 1. Initialization and Model Loading

```cpp
LightGlue::LightGlue(const std::string &model_path, bool use_cuda, bool use_fp16)
{
    // ...
    std::string engine_path = model_path;
    size_t ext_pos = engine_path.find_last_of(".");
    if (ext_pos != std::string::npos) {
        engine_path = engine_path.substr(0, ext_pos) + ".engine";
    }

    try {
        mModel = std::make_shared<TRTModel>(engine_path);
        // ...
```
**Explanation:** 
- The constructor takes a path to a PyTorch model (`.pt`), but immediately rewrites the extension to `.engine`. 
- **TensorRT Optimization:** PyTorch models are too slow for real-time SLAM. The system assumes that an offline script has already compiled the LightGlue PyTorch model into a highly optimized TensorRT engine file, which strips Python overhead, fuses GPU kernels, and optionally drops precision to FP16.
- It loads this engine into a custom `TRTModel` object.

### 2. Preprocessing Data for Neural Networks

```cpp
torch::Tensor LightGlue::normalizeKeypoints(const std::vector<cv::KeyPoint> &kpts, const cv::Size &image_size)
{
    // ...
    for (size_t i = 0; i < kpts.size(); i++) {
        accessor[i][0] = (kpts[i].pt.x - w / 2.0f) / (max_size / 2.0f);
        accessor[i][1] = (kpts[i].pt.y - h / 2.0f) / (max_size / 2.0f);
    }
}
```
**Explanation:** 
- Neural networks, particularly the Positional Encoding layers in Transformers, require normalized inputs to function properly. 
- Pixel coordinates (which range from 0 to 1920, for example) are mathematically normalized to range from `-1.0` to `1.0`. The division by `max_size` preserves the image aspect ratio so the network doesn't distort the geometry.

### 3. The Core Matching Execution

```cpp
std::vector<LightGlueMatch> LightGlue::match(const std::vector<cv::KeyPoint> &kpts0, const cv::Mat &desc0, ...)
{
    std::lock_guard<std::mutex> lock(getInferenceMutex());
    // ...
```
**Explanation:** 
- **Thread Safety:** Neural network inference on the GPU must be strictly synchronized across threads. The `std::lock_guard` ensures that if Tracking and Loop Closing both try to run neural matching simultaneously, they don't corrupt the TensorRT execution context.

**Step 3a: Tensor Padding Strategy**
```cpp
    auto kpts0_raw = torch::empty({1 * n0 * 2 + 1024}, ...);
    auto kpts0_tensor = kpts0_raw.slice(0, 0, 1 * n0 * 2).reshape({1, (long)n0, 2});
```
- **Crucial GPU Hack:** When allocating CUDA tensors for the keypoints and descriptors, the code intentionally allocates an extra `1024` floats of raw memory, and then creates a `slice` to only view the valid data.
- **Why?** Transformer self-attention layers calculate similarities across all points. If there is a slight memory out-of-bounds read inside a highly optimized TensorRT CUDA kernel, it can read uninitialized garbage, resulting in `NaN` (Not a Number) values that catastrophically propagate through the whole matrix. Padding the buffer completely prevents this edge-case NaN corruption.

**Step 3b: TensorRT Execution**
```cpp
    auto context = mModel->getContext();
    context->setInputShape("kpts0", nvinfer1::Dims3{1, n0, 2});
    // ... set dynamic shapes ...
    context->setTensorAddress("kpts0", kpts0_tensor.data_ptr());
    // ... pass CUDA pointers ...

    cudaStream_t py_stream = at::cuda::getCurrentCUDAStream().stream();
    mModel->enqueue(py_stream);
    torch::cuda::synchronize();
```
- TensorRT is configured for **Dynamic Shapes**, meaning the number of keypoints (`n0`, `n1`) can change every frame. We tell the TensorRT context the exact dimensions of this specific frame.
- We directly pass the raw GPU memory pointers (`data_ptr()`) from the LibTorch tensors to the TensorRT context. This achieves **Zero-Copy Inference**: the data never leaves the GPU VRAM, ensuring maximum speed.
- Execution is dispatched asynchronously to the active CUDA stream (`enqueue`), and the CPU halts (`synchronize`) until the GPU finishes the matrix multiplications.

**Step 3c: Output Parsing**
```cpp
    matches0_tensor = matches0_tensor.cpu();
    match_scores_tensor = match_scores_tensor.cpu();

    for (int i = 0; i < n0; i++) {
        if (idx_acc[i] > -1) {
            LightGlueMatch m;
            m.idx0 = i;
            m.idx1 = idx_acc[i];
            m.score = score_acc[i];
            matches.push_back(m);
        }
    }
```
- The resulting match indices and confidence scores are pulled off the GPU back to the CPU RAM.
- If `idx_acc[i] > -1`, it means LightGlue successfully found a match for point `i` in the first image, mapping it to point `idx_acc[i]` in the second image. These are parsed into lightweight `LightGlueMatch` structs and returned to the SLAM pipeline.
