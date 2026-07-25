# Documentation: `G2oTypes.h`

## High-Level Overview
The `G2oTypes.h` file contains the custom vertex and edge definitions required to run **Bundle Adjustment (BA)** and **Pose Graph Optimization** using the `g2o` (General Graph Optimization) library.
`g2o` models SLAM as a massive factor graph where the **Vertices** (Nodes) represent the variables we want to optimize (e.g., Camera Poses, 3D MapPoints, IMU Biases, Gravity direction), and the **Edges** represent the physical or visual measurements that constrain those variables (e.g., "I observed this 3D point at this 2D pixel"). The optimizer iteratively shifts the Vertices to minimize the error along all the Edges simultaneously.

**Primary Dependencies:**
- `Thirdparty/g2o/...` (The g2o graph optimization backend).
- `Eigen` (For all the heavy matrix math and Jacobians).

---

## Block-by-Block Breakdown

### 1. Lie Algebra Math Helpers

```cpp
Eigen::Matrix3d ExpSO3(const double x, const double y, const double z);
Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R);
Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v);
```
**Explanation:** 
- In SLAM, optimizing 3D rotations is notoriously difficult because standard matrices suffer from "Gimbal Lock" and standard addition breaks orthogonality ($R_{new} \neq R_{old} + \Delta R$).
- **Lie Algebra ($\mathfrak{so}(3)$):** These functions map rotations from the Lie Group $SO(3)$ (a $3 \times 3$ Rotation Matrix) to the Lie Algebra $\mathfrak{so}(3)$ (a 3D vector representing an axis-angle rotation) using the Matrix Logarithm (`LogSO3`). 
- Optimization math is performed cleanly in the vector space, and then mapped back to a Rotation Matrix using the Matrix Exponential (`ExpSO3`).
- The Jacobians provided here are the exact mathematical derivatives of these mappings, which g2o needs to calculate the gradients during optimization.

### 2. Vertices (The State Variables)

```cpp
class VertexPose : public g2o::BaseVertex<6, ImuCamPose>
class VertexVelocity : public g2o::BaseVertex<3, Eigen::Vector3d>
class VertexGyroBias : public g2o::BaseVertex<3, Eigen::Vector3d>
```
**Explanation:** 
- `g2o` requires you to inherit from `BaseVertex<D, T>` where `D` is the degrees of freedom (DoF) and `T` is the underlying data type.
- `VertexPose`: Represents a camera pose with 6 DoF (3 for translation, 3 for rotation via $\mathfrak{so}(3)$).
- `VertexVelocity`: Represents the drone's 3D linear velocity ($v_x, v_y, v_z$).
- `VertexGyroBias` / `VertexAccBias`: Represent the slight, drifting electrical biases inherent to the IMU sensor. By adding these as vertices, the SLAM system can actually "learn" and calibrate the IMU drift in real-time.

```cpp
virtual void oplusImpl(const double* update_)
```
- Every vertex must implement `oplusImpl`. This defines exactly how a calculated update step ($\Delta x$) is applied to the current estimate ($x_{new} = x_{old} \oplus \Delta x$). For velocities, this is simple addition. For poses, this involves Lie Algebra multiplication.

### 3. Edges (The Measurements / Constraints)

```cpp
class EdgeMono : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, VertexPose>
```
**Explanation:** 
- `BaseBinaryEdge` connects exactly two vertices (a 3D MapPoint and a Camera Pose). 
- The measurement (`Eigen::Vector2d`) is the exact $(u,v)$ pixel where the neural network found the feature.
- `computeError()`: The core of the optimizer. It projects the 3D MapPoint into the Camera Pose, generating a predicted $(u,v)$ pixel. The `_error` is the difference between the prediction and the actual neural network measurement.

```cpp
class EdgeInertial : public g2o::BaseMultiEdge<9, Vector9d>
```
**Explanation:** 
- This is a massively complex edge connecting 5 different vertices: Pose $i$, Velocity $i$, Pose $j$, Velocity $j$, and the IMU Biases.
- It represents the physical IMU constraint: "Based on my accelerometer and gyroscope readings between frame $i$ and frame $j$, this is how far I *should* have moved."
- It compares the IMU integration against the purely visual pose estimates, forcing the vision and the physics to agree.

### 4. Analytical Jacobians

```cpp
virtual void linearizeOplus();
Eigen::Matrix<double,2,9> GetJacobian();
```
**Explanation:** 
- For every edge, the developer *could* let g2o compute the derivatives using numerical finite differences (e.g., wiggle the point by 0.0001 and see how the error changes). However, that is extremely slow.
- `linearizeOplus()` provides the exact, hard-coded analytical calculus derivatives (Jacobians) for every single measurement relative to every single state variable. This is what allows `SP_SLAM3` to optimize thousands of variables in real-time.
