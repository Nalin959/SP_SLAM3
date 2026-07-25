# Documentation: `OptimizableTypes.cpp`

## High-Level Overview
The `OptimizableTypes.cpp` file contains custom mathematical definitions used by **g2o** (General Graph Optimization), which is the backend optimization library powering ORB-SLAM3.
SLAM is fundamentally an optimization problem: we have noisy measurements (2D pixels in an image) and we want to find the most likely 3D geometry (MapPoints) and 6-DOF camera poses (KeyFrames) that produced those pixels. This is called Bundle Adjustment (BA). 
This file defines the **Edges** (the error/cost functions) that connect **Vertices** (the variables we are trying to optimize). Specifically, it provides the analytical Jacobians (derivatives) for these error functions, which allows g2o's Levenberg-Marquardt algorithm to converge extremely quickly.

**Primary Dependencies:**
- `OptimizableTypes.h`
- `g2o` library components (`BaseBinaryEdge`, `VertexSE3Expmap`, `VertexSBAPointXYZ`, `VertexSim3Expmap`).
- Custom geometric camera models (`GeometricCamera`).

---

## Block-by-Block Breakdown

### 1. Motion-Only Pose Optimization (`EdgeSE3ProjectXYZOnlyPose`)

```cpp
void EdgeSE3ProjectXYZOnlyPose::linearizeOplus() {
    g2o::VertexSE3Expmap * vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
    Eigen::Vector3d xyz_trans = vi->estimate().map(Xw);
    // ...
    Eigen::Matrix<double,3,6> SE3deriv;
    SE3deriv << 0.f, z,   -y, 1.f, 0.f, 0.f,
                 -z , 0.f, x, 0.f, 1.f, 0.f,
                 y ,  -x , 0.f, 0.f, 0.f, 1.f;

    _jacobianOplusXi = -pCamera->projectJac(xyz_trans) * SE3deriv;
}
```
**Explanation:** 
- This Edge is used in the **Tracking** thread. The 3D MapPoints (`Xw`) are assumed to be fixed (constant), and the system is *only* trying to optimize the 6-DOF pose of the current camera.
- `linearizeOplus()` computes the Jacobian of the reprojection error with respect to a small update $\delta\xi$ in the SE(3) lie algebra of the camera pose.
- `SE3deriv` is the standard generator matrix for the SE(3) Lie group, representing the derivative of a 3D point rotated and translated by the camera.
- It chain-rules this with `pCamera->projectJac(xyz_trans)`, which is the derivative of the specific camera projection model (e.g., Pinhole or Kannala-Brandt fish-eye).

### 2. Stereo/Right Camera Motion-Only Optimization (`EdgeSE3ProjectXYZOnlyPoseToBody`)

```cpp
void EdgeSE3ProjectXYZOnlyPoseToBody::linearizeOplus() {
    // ...
    Eigen::Vector3d X_r = mTrl.map(T_lw.map(Xw));
    // ...
    _jacobianOplusXi = -pCamera->projectJac(X_r) * mTrl.rotation().toRotationMatrix() * SE3deriv;
}
```
**Explanation:** 
- In a stereo rig, the system tracks points in both the left and right cameras. However, the system only optimizes a single "body" pose (the left camera).
- This edge represents a measurement in the *right* camera. The 3D point is transformed into the left camera (`T_lw`), then transformed from the left to the right camera using the fixed stereo extrinsic calibration (`mTrl`).
- The Jacobian is identical to the left camera, but chain-ruled through the fixed rotation matrix of the right camera (`mTrl.rotation()`).

### 3. Full Bundle Adjustment (`EdgeSE3ProjectXYZ`)

```cpp
void EdgeSE3ProjectXYZ::linearizeOplus() {
    g2o::VertexSE3Expmap * vj = static_cast<g2o::VertexSE3Expmap *>(_vertices[1]); // Camera Pose
    g2o::VertexSBAPointXYZ* vi = static_cast<g2o::VertexSBAPointXYZ*>(_vertices[0]); // 3D Point
    // ...
    _jacobianOplusXi =  projectJac * T.rotation().toRotationMatrix(); // Jacobian wrt 3D Point
    // ...
    _jacobianOplusXj = projectJac * SE3deriv; // Jacobian wrt Camera Pose
}
```
**Explanation:** 
- This Edge is used in the **Local Mapping** thread for Local Bundle Adjustment.
- Unlike the "OnlyPose" edges, here *both* the camera pose and the 3D MapPoint coordinate are variable and being optimized simultaneously.
- Therefore, this edge must calculate two Jacobians:
  1. `_jacobianOplusXi`: How does the 2D pixel error change if we move the 3D point? (This is simply the camera projection Jacobian rotated by the camera's orientation).
  2. `_jacobianOplusXj`: How does the 2D pixel error change if we move the camera? (This is identical to the OnlyPose math).

### 4. Sim3 Optimization (Loop Closing)

```cpp
bool VertexSim3Expmap::read(std::istream& is)
bool VertexSim3Expmap::write(std::ostream& os) const
```
**Explanation:** 
- When a loop is detected, monocular SLAM must correct 7 degrees of freedom: Rotation (3), Translation (3), and **Scale (1)**. This requires the `Sim3` Lie Group.
- This Vertex holds a 7-DOF Sim3 transformation.
- The `read` and `write` functions serialize/deserialize the 7 parameters (represented via the Lie algebra `.log()`) and the camera intrinsic parameters so they can be saved to disk.

```cpp
EdgeSim3ProjectXYZ::EdgeSim3ProjectXYZ()
EdgeInverseSim3ProjectXYZ::EdgeInverseSim3ProjectXYZ()
```
**Explanation:** 
- These are the error functions for the Sim3 optimizer used in `LoopClosing.cc`. 
- Note that they do *not* have a custom `linearizeOplus()` implemented in this file. When omitted, g2o will automatically fall back to computing the Jacobian using **numerical differentiation** (slightly tweaking the values to measure the slope). While slower than analytical Jacobians, Loop Closing runs relatively infrequently in the background, so the performance hit is acceptable in exchange for mathematical simplicity.
