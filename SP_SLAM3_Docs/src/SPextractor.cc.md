# Documentation: `SPextractor.cc`

## High-Level Overview
The `SPextractor.cc` file is responsible for extracting features from images. In the original ORB-SLAM3, this was done using the hand-crafted ORB (Oriented FAST and Rotated BRIEF) algorithm. This modified version (`SP_SLAM3`) completely replaces ORB with **SuperPoint** (a Deep Learning-based feature extractor).
The `SPextractor` class takes raw OpenCV images, builds an image pyramid (for scale invariance), feeds the images into the LibTorch/TensorRT SuperPoint model, and then rigorously enforces a uniform spatial distribution of the resulting keypoints using a custom **Quadtree (OctTree)** algorithm.

**Primary Dependencies:**
- `SPextractor.h`, `SuperPoint.h` (the neural network interface)
- `LightGlue.h` (for GPU inference mutexing)
- OpenCV for image manipulation (pyramids, resizing, borders).

---

## Block-by-Block Breakdown

### 1. Initialization and Parameter Setup

```cpp
SPextractor::SPextractor(int _nfeatures, float _scaleFactor, int _nlevels, ...)
```
**Explanation:** 
- The constructor allocates the structure for a multi-level image pyramid.
- It instantiates the `SPDetector` (the TensorRT/LibTorch SuperPoint wrapper) pointing to `"superpoint.engine"`.
- It calculates the scale factors and inverted scale factors for each level of the pyramid.
- **Feature Distribution per Level (`mnFeaturesPerLevel`):** It calculates exactly how many of the requested `nfeatures` should be extracted from each level of the pyramid. It uses a geometric progression so that the highest resolution (Level 0) gets the most features, and the lowest resolution gets the fewest.

### 2. The Image Pyramid

```cpp
void SPextractor::ComputePyramid(cv::Mat image)
```
**Explanation:** 
- SuperPoint, unlike some newer networks (like ALIKED), is not inherently scale-invariant. To make it work in SLAM (where cameras move forward and backward), the system must explicitly create an image pyramid.
- It iterates `nlevels` times, resizing the image down by `scaleFactor` (typically 1.2x) at each step.
- `copyMakeBorder`: It adds an artificial padded border (`EDGE_THRESHOLD`) around the image using `BORDER_REFLECT_101`. This ensures that features near the edge of the screen can still have their descriptors computed accurately without falling out of bounds.

### 3. Neural Network Feature Extraction (`ComputeKeyPointsOctTree`)

```cpp
void SPextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints, cv::Mat &_desc)
```
**Explanation:** 
- This is the main processing engine. It iterates over every level of the image pyramid.
- **GPU Mutexing (`std::lock_guard<std::mutex> gpuLock(LightGlue::getInferenceMutex())`):** Because SuperPoint runs on the GPU, it must lock the global GPU inference mutex to prevent collisions with the `PlaceRecognition` or `LightGlue` threads.
- `model->detect(...)`: It pushes the pyramid level into the neural network to get raw keypoints.
- **Grid-based Sub-extraction:** It conceptually divides the image into a $W \times W$ grid (typically $30 \times 30$ pixels) and extracts points from each cell independently. This guarantees that features are found everywhere in the image, not just clustered in high-contrast areas. If the default threshold (`iniThFAST`) yields no points in a cell, it lowers the threshold (`minThFAST`) and tries again.
- **Quadtree Distribution:** It takes the raw, messy list of keypoints and passes it to `DistributeOctTree`.
- `model->computeDescriptors(...)`: Once the final spatially-distributed keypoints are selected, it queries the SuperPoint neural network again to extract the 256-dimensional float descriptors exactly at those pixel coordinates.
- **Descriptor Formatting:** It strictly enforces that the output descriptors are a dense $N \times 256$ matrix of type `CV_32F`.

### 4. Quadtree Spatial Distribution (`DistributeOctTree`)

```cpp
vector<cv::KeyPoint> SPextractor::DistributeOctTree(...)
```
**Explanation:** 
- Neural networks often cluster points around highly textured objects (like a keyboard) and ignore blank walls. A SLAM system will fail if all its tracking points are in one corner of the screen. This function forces the points to spread out uniformly.
- **Initialization:** It divides the screen into an initial set of large rectangular `ExtractorNode`s.
- **Subdivision Loop:** 
  - It iterates through all active nodes.
  - If a node has more than one keypoint inside it, it calls `DivideNode()`, splitting the node into 4 smaller child quadrants (UL, UR, BL, BR) and re-assigning the keypoints to the children.
  - If a node is empty, it is deleted.
  - If a node has exactly one keypoint, it marks itself as `bNoMore` (finished).
- **Termination:** This loop continues subdividing until the number of active nodes exactly equals the requested number of features (`N`), or until no nodes can be subdivided further.
- **Selection:** Finally, it iterates through the surviving leaf nodes. If a node contains multiple keypoints, it selects the one with the highest neural network `response` score. It discards the rest.

### 5. The Functor Interface

```cpp
void SPextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints, OutputArray _descriptors)
```
**Explanation:** 
- This overloads the `()` operator so the `SPextractor` object can be called like a function. It serves as the top-level API for the `Tracking` thread.
- It takes the raw image, optionally builds the pyramid, calls the OctTree extraction, and finally formats the output `_keypoints` and `_descriptors` arrays to be consumed by the SLAM pipeline. It also re-scales the keypoint coordinates from the lower pyramid levels back to the original Level 0 image scale.
