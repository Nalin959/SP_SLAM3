# Documentation: `Optimizer.h`

## High-Level Overview
The `Optimizer.h` file acts as the primary interface between the high-level SLAM logic and the low-level **g2o** (General Graph Optimization) math engine.
It is an entirely static class containing dozens of highly specialized Bundle Adjustment (BA) configurations. Whenever any thread (Tracking, LocalMapping, LoopClosing) needs to refine a trajectory or a point cloud, it calls one of the functions in this class. The `Optimizer` constructs the specific g2o factor graph (Vertices and Edges), configures the sparse linear solver (usually Levenberg-Marquardt), executes the optimization, and pushes the optimized numbers back into the SLAM data structures.

**Primary Dependencies:**
- `Thirdparty/g2o/*` (The heavy backend solver).
- `Map.h`, `KeyFrame.h`, `MapPoint.h` (The entities being optimized).

---

## Block-by-Block Breakdown

### 1. The Bundle Adjustment Suite

```cpp
void static BundleAdjustment(...);
void static GlobalBundleAdjustemnt(...);
void static LocalBundleAdjustment(...);
void static MergeBundleAdjustmentVisual(...);
```
**Explanation:** 
- **Bundle Adjustment (BA)** is the gold standard for visual SLAM optimization. It jointly optimizes camera poses and 3D point positions to minimize the reprojection error.
- `LocalBundleAdjustment`: Used constantly by the `LocalMapping` thread. It optimizes a small sliding window of recent KeyFrames to keep the active edge of the map highly accurate without wasting CPU on old data.
- `GlobalBundleAdjustemnt`: Used exclusively by the `LoopClosing` thread. It optimizes *every single pose and point* in the entire map. This is incredibly slow (taking seconds or minutes) but yields mathematically perfect results.

### 2. Fast Pose Optimization

```cpp
int static PoseOptimization(Frame* pFrame);
int static PoseInertialOptimizationLastFrame(Frame *pFrame, ...);
```
**Explanation:** 
- These are "Motion-Only" optimizations used exclusively by the `Tracking` thread running at 30-60 FPS.
- Instead of optimizing everything, these functions lock the 3D MapPoints in place (assuming they are perfect) and *only* optimize the 6-DOF camera pose of the current live `Frame`. Because the problem is so small, it solves in less than a millisecond.

### 3. Loop Closing Solvers

```cpp
void static OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF, ...);
static int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2, ...);
```
**Explanation:** 
- When a loop is detected, `OptimizeSim3` is called first. It figures out the 7-DOF similarity transformation (Rotation, Translation, Scale) between the current frame and the historical loop frame.
- `OptimizeEssentialGraph`: Once the gap is measured, this function propagates the correction backward through the map. Instead of full Bundle Adjustment, it performs Pose Graph Optimization (PGO) on the Essential Graph (Spanning Tree). It warps the camera trajectory into place but does *not* optimize the 3D points, making it fast enough to prevent the drone from crashing while it waits.

### 4. Inertial Initialization Solvers

```cpp
void static InertialOptimization(Map *pMap, Eigen::Matrix3d &Rwg, double &scale, Eigen::Vector3d &bg, Eigen::Vector3d &ba, ...);
void static FullInertialBA(...);
```
**Explanation:** 
- Visual-Inertial SLAM is fundamentally a massive non-linear estimation problem. 
- `InertialOptimization` is used during the first 2-5 seconds of flight. It analyzes the visual trajectory and the IMU readings to mathematically deduce the direction of Gravity (`Rwg`), the true metric size of the world (`scale`), and the factory errors of the IMU chip (`bg`, `ba`).
- `FullInertialBA`: The ultimate solver. Once initialized, this function jointly optimizes everything at once: Camera Poses, 3D Points, IMU Velocities, and IMU Biases, tightly coupling the vision and inertial sensors into a single flawless trajectory.
