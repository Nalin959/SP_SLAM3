# Documentation: `Frame.h`

## High-Level Overview
The `Frame.h` file defines the `Frame` class, which is arguably the most frequently instantiated object in the entire SLAM system.
A `Frame` encapsulates a single snapshot in time from the sensors (Monocular, Stereo, or RGB-D, plus the IMU). It does not hold the raw image pixels in memory for long; instead, it holds the *abstractions* of that image: the extracted SuperPoint features, the LightGlue descriptors, the estimated camera pose ($T_{cw}$), and pointers to the 3D `MapPoint`s that correspond to those 2D features.

Because a SLAM system processes 30+ frames per second, this class is heavily optimized. Features are partitioned into a spatial grid to accelerate feature-matching, and images are immediately discarded once features are extracted to save RAM.

**Primary Dependencies:**
- `SPextractor.h` (Neural Network feature extraction).
- `ImuTypes.h` (Pre-integrated IMU measurements up to this frame).
- `GeometricCamera.h` (To unproject 2D pixels into 3D rays).
- `DBoW3` (Bag of Words vectors for loop closure and relocalization).

---

## Block-by-Block Breakdown

### 1. Constructors for Sensor Types

```cpp
Frame(const cv::Mat &imLeft, const cv::Mat &imRight, ...); // Stereo
Frame(const cv::Mat &imGray, const cv::Mat &imDepth, ...); // RGB-D
Frame(const cv::Mat &imGray, ...);                         // Monocular
```
**Explanation:** 
- The system supports multiple sensor paradigms. Each constructor is tailored to the inputs.
- Inside the constructor, the frame immediately invokes the `SPextractor` on the image to populate `mvKeys` (keypoints) and `mDescriptors` (deep learning feature vectors).
- **RGB-D/Stereo Depth:** For Stereo and RGB-D, the constructor computes the physical depth ($Z$) of every keypoint immediately and stores it in `mvDepth`. This allows instant 3D triangulation. Monocular cameras cannot do this and must wait for a second frame to triangulate depth.

### 2. Core State Variables (Pose & Calibration)

```cpp
cv::Mat mTcw;            // Camera pose (World to Camera transformation)
cv::Mat mRwc, mOw;       // Inverse rotation and Camera Center in world coordinates
GeometricCamera* mpCamera;
cv::Mat mK, mDistCoef;
```
**Explanation:** 
- `mTcw` is a $4 \times 4$ transformation matrix representing the pose of the camera in the global World coordinate frame. 
- `mOw` is the origin (center) of the camera in world space.
- The `GeometricCamera` pointer allows this specific frame to accurately unproject its 2D pixels into 3D rays, regardless of whether it was captured with a pinhole or fisheye lens.

### 3. Feature and Map Associations

```cpp
int N;
std::vector<cv::KeyPoint> mvKeysUn;
cv::Mat mDescriptors;
std::vector<MapPoint*> mvpMapPoints;
std::vector<bool> mvbOutlier;
```
**Explanation:** 
- `N`: Total number of features extracted by SuperPoint.
- `mvKeysUn`: Undistorted 2D pixel coordinates of the features.
- `mDescriptors`: The dense vector embeddings for each feature.
- **The Core Link:** `mvpMapPoints` is an array of size $N$. If feature index $i$ corresponds to a physical 3D point in the global map, `mvpMapPoints[i]` points to that `MapPoint`. If it's a new, untracked feature, the pointer is `NULL`.
- `mvbOutlier`: Flags if the robust optimizer (g2o) has determined that a specific 2D-3D match is statistically impossible (an outlier).

### 4. Grid-based Acceleration

```cpp
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64
std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, ...) const;
```
**Explanation:** 
- To track features between frames, the system needs to find matching points. Doing a brute-force $N \times N$ comparison of all features is mathematically $O(N^2)$ and too slow.
- Instead, the image is divided into a $64 \times 48$ grid. Each cell in `mGrid` contains the indices of the features that fall into that physical space.
- `GetFeaturesInArea`: When the system wants to track a feature at $(x, y)$, it only searches the grid cells within a small radius $r$. This reduces the search complexity from $O(N)$ to $O(1)$ constant time per feature, ensuring real-time performance.

### 5. IMU Integration

```cpp
cv::Mat mVw;                         // Linear velocity
IMU::Bias mImuBias;                  // Gyroscope and Accelerometer bias
IMU::Preintegrated* mpImuPreintegrated;
```
**Explanation:** 
- If running Visual-Inertial SLAM, every `Frame` also carries a state vector for physics.
- `mVw`: The linear velocity of the drone at the exact moment this image was captured.
- `mpImuPreintegrated`: Between the previous frame at $t-1$ and this frame at $t$, the IMU may have recorded 20 high-frequency samples (since IMUs run much faster than cameras). `mpImuPreintegrated` mathematically bundles all 20 of those IMU readings into a single constraint that ties this frame's physics to the previous frame's physics.
