# Documentation: `SPmatcher.h`

## High-Level Overview
The `SPmatcher.h` file acts as the grand junction for all feature matching operations in the SLAM system.
In a visual SLAM system, discovering how pixels in Frame A relate to pixels in Frame B is the single most important (and computationally expensive) task. `SPmatcher` handles this by providing dozens of highly specialized search functions depending on *why* the system is searching (e.g., Tracking vs. Triangulation vs. Loop Closing).
Crucially, in `SP_SLAM3`, this class seamlessly bridges the gap between classical geometric projections (Epipolar lines) and deep learning data association (`LightGlue`).

**Primary Dependencies:**
- `LightGlue.h` (The deep learning matcher).
- `Frame.h`, `KeyFrame.h`, `MapPoint.h` (The entities being matched).

---

## Block-by-Block Breakdown

### 1. Neural Network Integration

```cpp
static void SetLightGlue(std::shared_ptr<LightGlue> pLG);
static std::shared_ptr<LightGlue> GetLightGlue();
static float DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
```
**Explanation:** 
- The `LightGlue` neural network runs on the GPU and is incredibly heavy. We cannot afford to have 5 different threads each instantiate their own copy of the model in VRAM. 
- `SetLightGlue` creates a static, globally shared pointer to a single LightGlue instance, ensuring safe, memory-efficient inference across all SLAM threads.
- `DescriptorDistance`: In classic ORB-SLAM3, this computed the Hamming distance between binary strings. Here, it computes the L2 norm (Euclidean distance) or Cosine Similarity between the floating-point SuperPoint tensors.

### 2. Search by Projection (Tracking)

```cpp
int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);
int SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th=3);
```
**Explanation:** 
- Used by the high-speed `Tracking` thread. 
- If the drone was at $(x,y,z)$ a fraction of a second ago, physics dictates it cannot have teleported very far. `SearchByProjection` takes the 3D MapPoints seen in the *last* frame, uses a constant-velocity physics model to guess where the drone is *now*, and projects those 3D points onto the 2D image plane of the *current* frame.
- It then searches only a tiny, constrained pixel radius (`th`) around that projection for a matching feature. This is thousands of times faster than searching the whole image.

### 3. Search by Bag-of-Words (Relocalization / Loops)

```cpp
int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);
int SearchByBoW(KeyFrame *pKF1, KeyFrame* pKF2, std::vector<MapPoint*> &vpMatches12);
```
**Explanation:** 
- Used when the physical distance between cameras is huge or unknown (e.g., Loop Closing or Relocalization). We cannot project points because we don't know the camera pose.
- Instead, this function uses the Bag-of-Words vectors. If Feature A in Image 1 and Feature B in Image 2 belong to the exact same visual vocabulary "word" node, they are compared. This acts as a massive pre-filter before handing the surviving matches over to LightGlue.

### 4. Search for Triangulation (Local Mapping)

```cpp
int SearchForTriangulation(KeyFrame *pKF1, KeyFrame* pKF2, cv::Mat F12, ...);
```
**Explanation:** 
- Used by the `LocalMapping` thread to spawn new 3D points. 
- It takes two `KeyFrame`s and the Fundamental Matrix (`F12`) between them. It searches for matches, but explicitly rejects any match that does not lie perfectly on the Epipolar Line computed from `F12`. This geometric constraint prevents hallucinated or false-positive 3D points from corrupting the map.

### 5. Fusing Duplicates

```cpp
int Fuse(KeyFrame* pKF, const vector<MapPoint *> &vpMapPoints, const float th=3.0);
```
**Explanation:** 
- Used during map optimization. If the system projects a known MapPoint into a KeyFrame and finds that the KeyFrame *already* has a different MapPoint at that exact pixel, it realizes they are physical duplicates. `Fuse` triggers the logic to merge the two MapPoints into one.
