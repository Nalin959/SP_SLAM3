# Documentation: `Sim3Solver.h`

## High-Level Overview
The `Sim3Solver.h` file implements the core mathematical algorithm for solving a **Similarity Transformation ($Sim(3)$)** between two 3D coordinate frames.
In rigid body mechanics (like PnP or ICP), you usually solve for a 6-DOF transformation: $SE(3)$ (Rotation and Translation). However, pure Monocular SLAM suffers from *Scale Drift*. Over long distances, the camera might think the world is getting slightly bigger or slightly smaller because it has no absolute depth reference.
When the `LoopClosing` thread realizes the drone has returned to the start, it cannot just compute the $SE(3)$ distance. It must compute the $Sim(3)$ transformation (7-DOF: Rotation, Translation, and **Scale**). This class calculates that $Sim(3)$ matrix, which is then used to un-warp the scale drift across the entire map.

**Primary Dependencies:**
- OpenCV (Matrix math).
- `KeyFrame.h`, `MapPoint.h` (The data being aligned).

---

## Block-by-Block Breakdown

### 1. Initialization and Configuration

```cpp
Sim3Solver(KeyFrame* pKF1, KeyFrame* pKF2, const std::vector<MapPoint*> &vpMatched12, const bool bFixScale = true, ...);
void SetRansacParameters(double probability, int minInliers, int maxIterations);
```
**Explanation:** 
- The constructor accepts two `KeyFrame`s that are suspected to be a loop, and a list of `MapPoint` matches between them (found via `SPmatcher::SearchByBoW`).
- `bFixScale`: A critical flag. If the drone is flying with an IMU or a Stereo camera, scale is physically observable and absolute. In those cases, `bFixScale` is true, and the solver falls back to computing standard $SE(3)$ without stretching the map. If pure Monocular, it is false, enabling the 7th DOF.
- The solver is wrapped in RANSAC to handle outlier matches from the neural network.

### 2. The Horn Algorithm Implementation

```cpp
void ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C);
void ComputeSim3(cv::Mat &P1, cv::Mat &P2);
```
**Explanation:** 
- This class implements the classic **Horn's Method** for Absolute Orientation.
- `ComputeCentroid`: First, the algorithm computes the 3D center of mass (centroid) of the point cloud seen by `KF1` and the point cloud seen by `KF2`. It then shifts both point clouds so their centroids sit exactly at the origin $(0,0,0)$.
- `ComputeSim3`: With the translation temporarily eliminated by the centroid shift, it computes the cross-covariance matrix between the two point clouds. By extracting the Eigenvectors of this matrix (or using SVD), it solves directly for the Rotation (a Quaternion) and the Scale factor. Finally, it uses the rotation and scale to solve for the remaining Translation.

### 3. Execution and Refinement

```cpp
cv::Mat find(std::vector<bool> &vbInliers12, int &nInliers);
cv::Mat iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers);
```
**Explanation:** 
- `iterate`: The RANSAC loop. It selects 3 random 3D-to-3D matches (the absolute mathematical minimum required to solve $Sim(3)$), runs `ComputeSim3`, and counts how many of the other matched points agree with that solution (inliers).
- `CheckInliers`: Evaluates the current hypothesis by projecting the 3D points from `KF1` into the camera of `KF2` using the guessed $Sim(3)$ matrix. If the projected pixels land very close to the actual detected pixels in `KF2`, they are counted as inliers.

### 4. Output Accessors

```cpp
cv::Mat GetEstimatedRotation();
cv::Mat GetEstimatedTranslation();
float GetEstimatedScale();
```
**Explanation:** 
- Once the RANSAC loop finds a winning hypothesis, the system can extract the exact parameters.
- `GetEstimatedScale()` is particularly vital. If it returns $1.1$, it means the map drifted by $10\%$ and the `LoopClosing` thread must instruct the `Optimizer` to shrink the entire map back down before merging the loop seam.
