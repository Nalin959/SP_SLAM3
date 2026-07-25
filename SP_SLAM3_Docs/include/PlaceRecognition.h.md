# Documentation: `PlaceRecognition.h`

## High-Level Overview
The `PlaceRecognition.h` file introduces a **Deep Learning Global Descriptor Database**, completely replacing the traditional DBoW (Bag-of-Words) system for loop closure in this modified version of `SP_SLAM3`.
Instead of relying on handcrafted visual words to recognize if the drone has returned to a previously mapped location, this class uses a Convolutional Neural Network (such as NetVLAD or CosPlace). The CNN compresses an entire camera image down into a single, high-dimensional floating-point vector (the "Global Descriptor"). 
This makes loop closures incredibly resilient against extreme lighting changes, season changes, or severe motion blur—conditions where traditional feature-based BoW fails completely.

**Primary Dependencies:**
- `LibTorch` (C++ PyTorch API for running the CNN).
- `KeyFrame.h` (The entities being fingerprinted and stored).

---

## Block-by-Block Breakdown

### 1. Model Initialization

```cpp
PlaceRecognition(const std::string &model_path, bool use_cuda = true, bool use_fp16 = false);
torch::jit::script::Module mModel;
```
**Explanation:** 
- The constructor loads a pre-trained PyTorch JIT model (`.pt` or `.ptl` file) via LibTorch.
- Like the `LightGlue` class, it supports GPU acceleration (`use_cuda`) and Half-Precision (`use_fp16`) to ensure the neural network runs fast enough not to bottleneck the `LoopClosing` thread on edge devices (e.g., Jetson Orin NX).

### 2. Feature Extraction

```cpp
torch::Tensor preprocessImage(const cv::Mat &image);
torch::Tensor extractDescriptor(const cv::Mat &image);
```
**Explanation:** 
- `preprocessImage`: Resizes the raw OpenCV image, converts it from BGR to RGB, normalizes the pixel intensities to exactly what the PyTorch model was trained on (usually ImageNet stats), and moves it to the GPU.
- `extractDescriptor`: Feeds the preprocessed tensor through the CNN (`mModel.forward`) and returns the resulting 1D Global Descriptor tensor (typically 256 to 4096 dimensions depending on the architecture).

### 3. The Descriptor Database

```cpp
struct Entry {
    KeyFrame* pKF;
    torch::Tensor descriptor;  // [D] normalized
};
std::vector<Entry> mvDatabase;
```
**Explanation:** 
- This replaces the `KeyFrameDatabase` inverted file index. 
- Every time a new `KeyFrame` is created by the `LocalMapping` thread, its global descriptor is calculated and pushed into `mvDatabase` via the `add()` function. 
- The descriptors are structurally normalized so that mathematical distance can be computed using a simple dot product.

### 4. Querying for Loops

```cpp
std::vector<KeyFrame*> query(KeyFrame *pKF, const torch::Tensor &descriptor, int nCandidates, const std::set<KeyFrame*> &spConnectedKFs);
```
**Explanation:** 
- Called continuously by the `LoopClosing` thread. 
- It takes the global descriptor of the live `KeyFrame` and compares it against *every single entry* in `mvDatabase` using Cosine Similarity (achieved via batched PyTorch matrix multiplication on the GPU).
- **Crucial Detail:** `spConnectedKFs` contains the local neighbors of the current KeyFrame. The query function explicitly ignores these. If the drone is hovering in place, the current frame will obviously look identical to the frame from 1 second ago. Loop closing is only interested in matches that are far away in time/graph-distance, indicating a true loop. It returns the top `nCandidates` scoring historical KeyFrames.
