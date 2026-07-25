# Documentation: `Frame.cc`

## High-Level Overview
The `Frame.cc` file implements the `Frame` class, which is arguably the most central data structure in the SP_SLAM3 tracking pipeline. A `Frame` represents a single captured image (or stereo pair) at a specific timestamp. 
It encapsulates everything about that moment in time: the raw image data, the extracted SuperPoint/ORB features, the camera intrinsic model, IMU preintegration state, and the mathematical pose (`Tcw`) of the camera in the global map.

**Primary Dependencies:**
- `Frame.h`, `KeyFrame.h`, `MapPoint.h`
- `GeometricCamera.h`, `Pinhole.h`, `KannalaBrandt8.h`
- `SPextractor.h` (SuperPoint extraction logic)
- `IMU` and `g2o` mathematical types.

---

## Block-by-Block Breakdown

### 1. Frame Constructors

```cpp
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ... )
```
**Explanation:** 
- The class features multiple overloaded constructors depending on the sensor setup: **Stereo**, **RGB-D**, or **Monocular**.
- **Initialization:** Upon creation, it assigns itself a unique, sequentially increasing ID (`nNextId++`). 
- **Feature Extraction:** It immediately spawns two `std::thread` workers to run `ExtractORB` (or SuperPoint extraction) in parallel on the left and right images.
- **Undistortion & Matching:** After extraction, it calls `UndistortKeyPoints()` to mathematically flatten fisheye/lens distortion on the 2D keypoints, followed by `ComputeStereoMatches()` to find disparity between the left and right cameras to estimate depth.
- **Grid Assignment:** It divides the image into a spatial grid (`AssignFeaturesToGrid()`) to dramatically speed up 2D neighborhood searches later.

### 2. Spatial Grid Indexing

```cpp
void Frame::AssignFeaturesToGrid()
{
    const int nCells = FRAME_GRID_COLS*FRAME_GRID_ROWS;
    int nReserve = 0.5f*N/(nCells);
    // ...
    for(int i=0;i<N;i++) {
        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY)){
            mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }
}
```
**Explanation:** 
- **Design Pattern (Spatial Hashing):** Searching for matching points across an entire 1920x1080 image is computationally expensive ($O(N)$ per point). Instead, the image is binned into a 64x48 grid (`mGrid`). 
- Each extracted keypoint calculates which grid cell it falls into and stores its index there.
- Later, when searching for a feature near pixel `(x, y)`, the system only iterates through the points stored in the directly adjacent grid cells, reducing lookup time to $O(1)$.

```cpp
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel, const bool bRight) const
```
**Explanation:** 
- Leverages the spatial grid populated above. Given a target pixel `(x,y)` and a search radius `r`, it identifies the overlapping grid cells (`nMinCellX` to `nMaxCellX`) and returns a fast vector of candidate feature indices, optionally filtering by scale pyramid level.

### 3. Pose Management and IMU Integration

```cpp
void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}
```
**Explanation:** 
- Sets the $T_{cw}$ matrix (Transformation from World to Camera frame).
- Automatically updates internal cached matrices: Rotation ($R_{cw}$, $R_{wc}$), Translation ($t_{cw}$), and the absolute Camera Center in world coordinates ($O_w = -R_{cw}^T t_{cw}$).
- Caching these matrices is a critical performance optimization, as they are queried thousands of times per frame during map point reprojection.

```cpp
void Frame::SetImuPoseVelocity(const cv::Mat &Rwb, const cv::Mat &twb, const cv::Mat &Vwb)
```
**Explanation:** 
- If the system is using an IMU (Inertial Measurement Unit), the pose of the camera is dictated by the IMU Body frame (`b`) combined with the fixed Camera-to-Body calibration extrinsic matrix (`Tcb`).
- This function mathematically derives the visual camera pose $T_{cw}$ from the IMU body pose $T_{wb}$.

### 4. Frustum Culling (Visibility Checking)

```cpp
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
```
**Explanation:** 
- Determines if a 3D MapPoint is currently visible inside the camera's field of view (the view frustum).
- **Algorithm (Frustum Culling):**
  1. Transforms the 3D point from World to Camera coordinates ($P_c = R_{cw} * P + t_{cw}$).
  2. Checks if depth `Z > 0` (must be in front of the camera).
  3. Projects the 3D point to a 2D pixel `(u,v)`. Checks if the pixel falls within the image bounds (`mnMinX` to `mnMaxX`).
  4. **Scale Invariance:** Checks if the camera is too close or too far from the point based on the scale pyramid level at which the point was originally created (`dist < minDistance || dist > maxDistance`).
  5. **Viewing Angle:** Checks if the camera is viewing the point from an extreme, oblique angle. A point seen head-on cannot be reliably matched from a 90-degree side profile (`viewCos < viewingCosLimit`).
- **Optimization:** If the point passes all checks, it caches the projected `(u,v)` coordinates and predicted scale level inside the `MapPoint` itself so the tracking thread doesn't have to recalculate it.

### 5. Distortion Handling

```cpp
void Frame::UndistortKeyPoints()
```
**Explanation:** 
- If standard pinhole distortion coefficients (`k1, k2, p1, p2`) are provided, this uses `cv::undistortPoints` to move the extracted keypoints from their raw, distorted pixel locations to mathematically idealized, straight-line perspective coordinates.

```cpp
bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
```
**Explanation:** 
- The exact inverse of the above. When trying to draw or locate a 3D map point on the raw, distorted image, it projects the 3D point to ideal 2D coordinates, and then manually applies the radial (`k1, k2, k3`) and tangential (`p1, p2`) distortion polynomial formulas to shift the point outward/inward to match the warped lens.
