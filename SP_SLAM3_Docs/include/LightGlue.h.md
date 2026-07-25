# Documentation: `LightGlue.h`

## High-Level Overview
The `LightGlue.h` file acts as the C++ wrapper for the **LightGlue Neural Network**, representing a massive architectural shift from standard ORB-SLAM3. 
Traditionally, ORB-SLAM3 extracts ORB features and matches them using classical computer vision techniques (e.g., Hamming distance between binary descriptors). This is fast but highly prone to failure in low-light, low-texture, or GPS-denied environments (like drone flights through dark forests).
`SP_SLAM3` replaces classical matching with LightGlue, a state-of-the-art deep learning feature matcher. This class wraps a pre-trained ONNX/TensorRT model and uses it to find robust correspondences between two sets of SuperPoint features, significantly boosting the drone's ability to initialize its map and survive challenging visual conditions.

**Primary Dependencies:**
- `LibTorch` (C++ PyTorch API for tensor math).
- `TRTModel.h` (TensorRT for highly optimized GPU inference).
- OpenCV (For managing `cv::KeyPoint` structures).

---

## Block-by-Block Breakdown

### 1. Data Structures

```cpp
struct LightGlueMatch {
    int idx0;       // keypoint index in image 0
    int idx1;       // keypoint index in image 1
    float score;    // match confidence
};
```
**Explanation:** 
- A simple data container returned by the neural network. 
- It states: "Feature index `idx0` in Frame 1 mathematically corresponds to feature index `idx1` in Frame 2, with a confidence of `score`."
- The `score` allows the SLAM system to threshold and reject weak matches before they pollute the Bundle Adjustment graph.

### 2. Class Constructor

```cpp
class LightGlue {
public:
    LightGlue(const std::string &model_path, bool use_cuda = true, bool use_fp16 = false);
```
**Explanation:** 
- The constructor loads the neural network weights from `model_path` (an `.onnx` or `.trt` file).
- `use_cuda` and `use_fp16` dictate whether to run the model on the GPU and whether to use half-precision floating point math. Half-precision math is absolutely critical for deploying this model on resource-constrained drone hardware like the Jetson Orin NX, as it doubles inference speed without significantly degrading accuracy.

### 3. Core Inference Function

```cpp
std::vector<LightGlueMatch> match(
    const std::vector<cv::KeyPoint> &kpts0, const cv::Mat &desc0,
    const std::vector<cv::KeyPoint> &kpts1, const cv::Mat &desc1,
    const cv::Size &image_size0, const cv::Size &image_size1);
```
**Explanation:** 
- This is the main workhorse of the class. It is called by the `SPmatcher` whenever the system needs to find correspondences between two frames.
- It takes in the 2D pixel coordinates (`kpts0`, `kpts1`) and their corresponding deep learning embeddings (`desc0`, `desc1`) produced by the SuperPoint extractor.
- Unlike classical matching which just compares descriptors, LightGlue uses an Attention mechanism (Transformer) that also considers the spatial layout of the keypoints. This is why `image_size` is required—the keypoints must be normalized into $[-1, 1]$ coordinates for the neural network.

### 4. Concurrency Management

```cpp
static std::mutex& getInferenceMutex() {
    static std::mutex mtx;
    return mtx;
}
```
**Explanation:** 
- The SLAM system is heavily multi-threaded (Tracking, Local Mapping, and Loop Closing all run simultaneously).
- However, they all share a single GPU. If two threads try to run TensorRT inference at the exact same microsecond, the GPU memory manager will crash.
- `getInferenceMutex` provides a global lock. Before any thread can call `match()`, it must lock this mutex, ensuring that neural network inference happens strictly sequentially across the entire application.

### 5. Private Tensor Helpers

```cpp
private:
    torch::Tensor normalizeKeypoints(const std::vector<cv::KeyPoint> &kpts, const cv::Size &image_size);
    torch::Tensor descriptorsToTensor(const cv::Mat &desc);
```
**Explanation:** 
- TensorRT operates strictly on raw memory buffers (Tensors), while the rest of the SLAM system uses OpenCV `cv::Mat` and `cv::KeyPoint`.
- These helper functions handle the data wrangling: copying memory from the CPU into the GPU, normalizing coordinate systems, and reshaping the data dimensions to match exactly what the LightGlue neural network architecture expects.
