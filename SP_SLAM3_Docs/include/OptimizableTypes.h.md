# Documentation: `OptimizableTypes.h`

## High-Level Overview
The `OptimizableTypes.h` file extends the **g2o** (General Graph Optimization) library by defining custom "Edges" (error functions) specifically tailored for `SP_SLAM3`.
g2o is the math engine that powers all the Bundle Adjustment in this SLAM system. In a factor graph, a "Vertex" represents a variable we want to optimize (like a camera pose or a 3D point), and an "Edge" represents an observation connecting them (e.g., "Camera A saw Point B at pixel $(u, v)$").
This file defines the mathematical error functions for projecting 3D points into 2D pixels across various camera models (Pinhole, Fisheye) and coordinate frames (World-to-Camera, World-to-Body).

**Primary Dependencies:**
- `g2o` (The graph optimization framework).
- `GeometricCamera.h` (The polymorphic camera models used to project the points).
- `Eigen` (Linear algebra).

---

## Block-by-Block Breakdown

### 1. Motion-Only Edges (Pose Optimization)

```cpp
class EdgeSE3ProjectXYZOnlyPose: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>
```
**Explanation:** 
- A `UnaryEdge` connects to only one Vertex. In this case, the `VertexSE3Expmap` (the 6-DOF camera pose).
- **Use Case:** Used exclusively by the `Tracking` thread. When tracking a new frame, the 3D MapPoints are assumed to be perfect and fixed. The only variable being optimized is the camera pose.
- `computeError()`: Takes the known 3D point (`Xw`), maps it into the camera's coordinate frame using the estimated camera pose (`v1->estimate().map(Xw)`), and projects it into a 2D pixel using the `pCamera->project` function. The error is the vector difference between where the math *thinks* the pixel should be and where the neural network *actually* detected it (`_measurement`).

### 2. Stereo/Body-Offset Edges

```cpp
class EdgeSE3ProjectXYZOnlyPoseToBody: public g2o::BaseUnaryEdge<...>
{
    g2o::SE3Quat mTrl;
}
```
**Explanation:** 
- Similar to the above, but used for Stereo or Multi-Camera setups.
- The `mTrl` variable represents the rigid physical offset (Translation + Rotation) between the Right Camera and the Left Camera (or the IMU Body frame).
- `computeError()`: First moves the point into the Left Camera frame (`v1->estimate()`), then applies the rigid physical offset (`mTrl`) to move it into the Right Camera frame, before finally projecting it to 2D.

### 3. Full Bundle Adjustment Edges (Pose + Point Optimization)

```cpp
class EdgeSE3ProjectXYZ: public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
```
**Explanation:** 
- A `BinaryEdge` connects two Vertices: a 3D point (`VertexSBAPointXYZ`) and a camera pose (`VertexSE3Expmap`).
- **Use Case:** Used by `LocalMapping` and `LoopClosing`. During Bundle Adjustment, both the camera trajectory *and* the 3D map points are noisy. The optimizer jiggles both the poses and the points simultaneously to minimize the global reprojection error.
- `computeError()`: Projects the estimated 3D point (`v2`) using the estimated camera pose (`v1`) and compares it to the observed 2D pixel.

### 4. Sim3 Edges (Loop Closing Scale Drift)

```cpp
class VertexSim3Expmap : public g2o::BaseVertex<7, g2o::Sim3>
class EdgeSim3ProjectXYZ : public g2o::BaseBinaryEdge<...>
```
**Explanation:** 
- In pure Monocular SLAM (no IMU, no Stereo), the absolute scale of the world is unobservable and will slowly drift over time (e.g., 1 meter at the start of the flight becomes 1.1 meters at the end).
- When closing a loop, the system cannot just use rigid 6-DOF ($SE(3)$) transformations; it must use 7-DOF ($Sim(3)$) transformations, which include Rotation, Translation, and **Scale**.
- `VertexSim3Expmap` represents a camera pose that can stretch/shrink the world. `EdgeSim3ProjectXYZ` computes the reprojection error while accounting for this 7th dimension of scaling.
