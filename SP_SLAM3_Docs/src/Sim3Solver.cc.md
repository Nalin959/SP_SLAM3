# Documentation: `Sim3Solver.cc`

## High-Level Overview
The `Sim3Solver.cc` file implements an algorithmic solver to find the **Sim3** (Similarity Transform: 3D Rotation, 3D Translation, and 1D Scale) between two KeyFrames.
In Monocular SLAM, scale is completely unobservable. As a drone flies, the map's internal scale will slowly drift. When a Loop Closure is detected (the drone recognizes where it started), the scale of the current KeyFrame might be 1.05x larger than the historical KeyFrame. To close the loop seamlessly, we cannot just use a rigid 6-DOF (SE3) transform; we must calculate exactly how much the scale drifted and apply a 7-DOF Sim3 transform to align the two maps.
This solver uses Horn's classic 1987 method ("Closed-form solution of absolute orientation using unit quaternions") embedded inside a RANSAC loop.

**Primary Dependencies:**
- `Sim3Solver.h`
- `Random.h` (for RANSAC minimal set selection)
- OpenCV (`cv::eigen`, `cv::SVD`) for matrix decomposition.

---

## Block-by-Block Breakdown

### 1. Initialization and RANSAC Setup

```cpp
Sim3Solver::Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2, const vector<MapPoint *> &vpMatched12, const bool bFixScale, ...)
void Sim3Solver::SetRansacParameters(...)
```
**Explanation:** 
- The solver is initialized with two KeyFrames (`pKF1`, `pKF2`) and a list of matching `MapPoint`s between them. These matches were typically just discovered by the `PlaceRecognition` / Bag-of-Words module.
- It extracts the 3D world coordinates of those MapPoints and transforms them into the local camera coordinates of their respective KeyFrames (`mvX3Dc1`, `mvX3Dc2`).
- **`bFixScale`:** If the system is running Stereo or RGB-D SLAM, scale is physically observable and does not drift. In this case, `bFixScale` is true, and the solver essentially degrades gracefully into a standard SE3 absolute orientation solver.
- **RANSAC:** Similar to `PnPsolver`, it calculates the maximum number of iterations required to guarantee finding a clean minimal set, adjusting `mRansacMaxIts` dynamically based on the requested probability.

### 2. The RANSAC Loop (`iterate`)

```cpp
cv::Mat Sim3Solver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    // ...
    for(short i = 0; i < 3; ++i) {
        // Randomly pick a minimal set (3 points for Sim3)
        mvX3Dc1[idx].copyTo(P3Dc1i.col(i));
        // ...
    }
    
    ComputeSim3(P3Dc1i, P3Dc2i);
    CheckInliers();
    
    if(mnInliersi>=mnBestInliers) {
        // Save best Sim3 transform
    }
}
```
**Explanation:** 
- To compute a 7-DOF Sim3 transformation, we mathematically require a minimum of **3 non-collinear 3D point matches**.
- Inside the loop, it randomly selects 3 indices, extracts their 3D coordinates from both camera frames, and passes them to the core algorithm (`ComputeSim3`).
- It then evaluates the resulting transformation against *all* other matches (`CheckInliers`).
- It keeps track of the Sim3 transformation that yields the highest number of inliers.

### 3. Horn's Absolute Orientation (`ComputeSim3`)

This function implements the core mathematics described in Horn's 1987 paper.

```cpp
void Sim3Solver::ComputeSim3(cv::Mat &P1, cv::Mat &P2)
```
**Explanation:** 
- **Step 1: Centroids.** It calculates the center of mass for both sets of 3 points (`O1`, `O2`). It then shifts all points so they are relative to their respective centroids (`Pr1`, `Pr2`). This completely decouples translation from rotation/scale.
- **Step 2 & 3: The Cross-Covariance Matrix (M) and Matrix (N).** 
  - It computes `M = Pr2 * Pr1.t()`. This matrix captures the cross-correlation between the two point clouds.
  - It constructs a $4 \times 4$ symmetric matrix `N` composed of sums and differences of the elements of `M`.
- **Step 4: Rotation (Eigen Decomposition).**
  - The beautiful mathematical trick of Horn's method is that the unit quaternion representing the optimal rotation between the point clouds is exactly the **Eigenvector corresponding to the largest Eigenvalue of N**.
  - `cv::eigen(N,eval,evec)` extracts this quaternion. It is converted to an Angle-Axis vector and then a standard $3 \times 3$ Rotation Matrix using `cv::Rodrigues`.
- **Step 6: Scale Calculation.**
  - If `mbFixScale` is false, it calculates the scale factor `ms12i` as the ratio of the rotated point cloud projections. (Conceptually, comparing the variance of the two point clouds).
- **Step 7: Translation Calculation.**
  - Finally, translation is trivially recovered by taking the centroid of Cloud 1, and subtracting the rotated, scaled centroid of Cloud 2.
- **Step 8: Matrix Assembly.**
  - It packages the Rotation, Translation, and Scale into two $4 \times 4$ matrices (`mT12i` and its inverse `mT21i`) which represent the full Sim3 transformation.

### 4. Evaluating Hypotheses (`CheckInliers`)

```cpp
void Sim3Solver::CheckInliers()
```
**Explanation:** 
- Takes the newly computed Sim3 transform (`mT12i`, `mT21i`).
- Takes *all* 3D points from Camera 2, transforms them into Camera 1, and projects them into 2D pixel coordinates (`vP2im1`).
- Takes *all* 3D points from Camera 1, transforms them into Camera 2, and projects them into 2D pixel coordinates (`vP1im2`).
- It calculates the bidirectional 2D reprojection error. If the error in *both* cameras is below the noise threshold (`mvnMaxError1`, `mvnMaxError2`), the match is officially an inlier for this specific Sim3 hypothesis.
