# Documentation: `TwoViewReconstruction.h`

## High-Level Overview
The `TwoViewReconstruction.h` file contains the mathematical engine used to initialize the entire SLAM system from scratch. 
When a monocular SLAM system boots up, it has no 3D map and no knowledge of its own trajectory. It is entirely blind. It must take two 2D images, find matches between them, and use pure projective geometry to hallucinate the initial 3D structure of the world and the relative motion between the two camera shots.
This class runs two geometric models simultaneously (a Fundamental Matrix and a Homography), mathematically scores which model better describes the scene, and then extracts the 3D points and 6-DOF camera pose.

**Primary Dependencies:**
- OpenCV (Matrix math and `cv::Point2f` / `cv::Point3f`).

---

## Block-by-Block Breakdown

### 1. The Main Reconstructor

```cpp
bool Reconstruct(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2, const std::vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);
```
**Explanation:** 
- The master function called by the `Tracking` thread during initialization.
- It is given two frames (`vKeys1`, `vKeys2`) and the list of LightGlue/SuperPoint matches between them.
- It launches two parallel `std::thread`s: one computes `FindHomography` and the other computes `FindFundamental`.
- Once both threads finish, it computes a heuristic score: $S_H / (S_H + S_F)$. If the score is $>0.40$, the scene is considered planar, and it initializes using the Homography. Otherwise, it initializes using the Fundamental Matrix.
- **Outputs:** The relative Rotation (`R21`) and Translation (`t21`), and the initial 3D point cloud (`vP3D`).

### 2. Geometric Model Computation

```cpp
void FindHomography(std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
void FindFundamental(std::vector<bool> &vbInliers, float &score, cv::Mat &F21);
cv::Mat ComputeH21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);
cv::Mat ComputeF21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);
```
**Explanation:** 
- Both functions run RANSAC loops. At each iteration, they sample a minimal set of points (4 points for `ComputeH21` using the DLT algorithm, 8 points for `ComputeF21` using the Normalized 8-Point algorithm).
- They then score the resulting matrix using `CheckHomography` or `CheckFundamental`, which counts how many of the other matches obey the geometric constraint (e.g., how close they are to the epipolar line).

### 3. Pose Extraction and Triangulation

```cpp
bool ReconstructF(std::vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K, cv::Mat &R21, cv::Mat &t21, ...);
bool ReconstructH(std::vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K, cv::Mat &R21, cv::Mat &t21, ...);
void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
```
**Explanation:** 
- A Fundamental Matrix ($F$) can be converted into an Essential Matrix ($E$) using the camera intrinsics ($K$). `DecomposeE` mathematically splits $E$ into 4 possible combinations of Rotation and Translation.
- Similarly, a Homography ($H$) can be decomposed into 8 possible combinations of Rotation, Translation, and Plane Normals (using Faugeras' SVD method).
- `ReconstructF` and `ReconstructH` test all possible combinations by triangulating the 3D points (`Triangulate()`) and checking which combination puts the majority of the 3D points *in front* of both cameras (positive depth). The combination with the most valid points is chosen as the true camera motion.
