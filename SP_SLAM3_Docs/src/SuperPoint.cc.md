# Documentation: `SuperPoint.cc`

## High-Level Overview
The `SuperPoint.cc` file implements the `SPDetector` class, which serves as the bridge between the C++ SLAM system and the Deep Learning SuperPoint model. 
This file handles loading the TensorRT engine, pushing images onto the GPU, executing the neural network forward pass, and pulling the resulting feature locations (keypoints) and their 256-dimensional float descriptors back to the CPU for the tracking and mapping threads to use.

**Primary Dependencies:**
- `SuperPoint.h`
- `TRTModel.h` (TensorRT inference wrapper)
- `LibTorch` / `ATen` for tensor manipulation (Softmax, NMS, Grid Sampling).
- `CUDA` for GPU stream synchronization.

---

## Block-by-Block Breakdown

### 1. Model Initialization

```cpp
SPDetector::SPDetector(const std::string& model_path, bool use_fp16)
```
**Explanation:** 
- The constructor accepts the path to a serialized TensorRT engine (e.g., `"superpoint.engine"`).
- It instantiates a `TRTModel`, which allocates the GPU memory contexts needed for inference.

### 2. Neural Network Forward Pass (`detect`)

```cpp
void SPDetector::detect(cv::Mat &img, bool use_cuda)
```
**Explanation:** 
- This function runs the actual Deep Learning inference. It operates in several strictly defined stages:
- **Preprocessing:** It converts the input image to Grayscale. Crucially, it applies **CLAHE (Contrast Limited Adaptive Histogram Equalization)**. This is specifically tuned to normalize contrast across different camera modalities (e.g., standard RGB, IR, and RAW footage), making the neural network robust to extreme lighting changes.
- **Tensor Upload:** It wraps the OpenCV `Mat` in a PyTorch Tensor, normalizes the pixels to `[0, 1]`, and uploads it to the GPU (`to(device)`). If `mbFP16` is enabled, it casts to half-precision.
- **TensorRT Execution:** It binds the input tensor and the output memory buffers (`mProb` for probabilities, `mDesc` for descriptors) to the TensorRT context. It executes the graph asynchronously on a CUDA stream (`enqueue`) and immediately calls `torch::cuda::synchronize()` to wait for completion.
- **Post-Processing (Softmax and Reshape):** The raw probability output from SuperPoint is a tensor of shape `[1, 65, H/8, W/8]`. The 65 channels represent an $8 \times 8$ grid of pixels (64) plus a "no feature" dustbin class (1). 
  - It applies a spatial `softmax` to turn raw logits into probabilities.
  - It strips off the dustbin class.
  - It uses PyTorch's highly optimized `view` and `permute` operations (the "Space-to-Depth" inverse operation) to unfold the $H/8 \times W/8 \times 64$ tensor back into a full-resolution $H \times W$ probability map.
- **Descriptor Normalization:** It L2-normalizes the raw descriptor tensor along the channel dimension so they can be matched using Cosine Similarity / L2 distance later.

### 3. Keypoint Extraction and Non-Maximum Suppression (NMS)

```cpp
void SPDetector::getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms)
```
**Explanation:** 
- This function extracts the actual X/Y pixel coordinates from the dense $H \times W$ probability map generated in `detect()`.
- **GPU-accelerated NMS:** If `nms` is true, it performs Non-Maximum Suppression directly on the GPU using PyTorch's `max_pool2d`. It sweeps a $9 \times 9$ window across the image and suppresses (zeroes out) any pixel that isn't the absolute maximum in its neighborhood. This ensures we don't extract 5 overlapping keypoints on the exact same corner.
- It thresholds the remaining probabilities to find valid keypoints.
- **Optimization:** It extracts the coordinates and the response scores while they are still on the GPU, and *only* transfers the final subset of valid points (`kpts_cpu`, `responses_cpu`) back over the PCIe bus to the CPU.

### 4. Descriptor Extraction (`computeDescriptors`)

```cpp
void SPDetector::computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
```
**Explanation:** 
- The SuperPoint neural network computes descriptors at a lower resolution ($H/8 \times W/8$). To get the descriptor for an exact sub-pixel keypoint location at full resolution, we must interpolate.
- **Grid Sampling (Bilinear Interpolation):** It packages the X/Y coordinates of the extracted keypoints into a normalized grid `[-1, 1]`.
- It uses PyTorch's `grid_sampler` function. This performs hardware-accelerated bilinear interpolation on the GPU, probing the $H/8 \times W/8$ descriptor map at exactly the right fractional coordinates to compute the 256D descriptor vector for each keypoint.
- It L2-normalizes the resulting vectors again, transfers them back to the CPU, and copies them into an OpenCV `cv::Mat` for the SLAM system to consume.

### 5. CPU Fallback NMS (`NMS2`, `NMS`)

*(Note: These functions are legacy or fallback implementations)*
**Explanation:** 
- These are CPU-based implementations of Non-Maximum Suppression using standard OpenCV grids. They iterate over the points, draw them into a grid, and suppress neighbors. They are vastly slower than the GPU MaxPool method used in `getKeyPoints`, and are generally bypassed in optimized TRT setups.
