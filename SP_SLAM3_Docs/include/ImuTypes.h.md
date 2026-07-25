# Documentation: `ImuTypes.h`

## High-Level Overview
The `ImuTypes.h` file provides the foundational data structures and mathematical operations required for **Visual-Inertial SLAM**.
While cameras capture the environment at 30 Hz, Inertial Measurement Units (IMUs) measure the drone's raw acceleration and angular velocity at very high frequencies (e.g., 200 Hz). This file implements the "IMU Preintegration" theory formulated by Forster et al. (2015), which elegantly bundles hundreds of high-frequency IMU measurements into a single mathematical constraint that can be seamlessly added to the g2o optimization graph alongside the visual camera data.

**Primary Dependencies:**
- `<Eigen/Core>`, `<Eigen/Geometry>` (For complex 3D math and Jacobians).
- `boost/serialization` (To save IMU state and biases to disk).

---

## Block-by-Block Breakdown

### 1. Basic IMU Data Structures

```cpp
class Point {
    cv::Point3f a; // Acceleration (m/s^2)
    cv::Point3f w; // Angular Velocity (rad/s)
    double t;      // Timestamp
};
```
**Explanation:** 
- A simple struct representing a single physical reading from the IMU hardware at a specific microsecond in time.

```cpp
class Bias {
    float bax, bay, baz; // Accelerometer Bias
    float bwx, bwy, bwz; // Gyroscope Bias
};
```
**Explanation:** 
- No IMU is perfect. Even when the drone is sitting perfectly still on a desk, the IMU will report a non-zero acceleration and rotation due to electrical noise and manufacturing imperfections. This offset is the **Bias**.
- If this bias is not mathematically subtracted from every measurement, the integration will drift exponentially, and the drone will think it is accelerating into outer space.
- SLAM treats this `Bias` as an optimizable variable, constantly recalculating it in real-time to track changes caused by temperature fluctuations on the drone's circuit board.

### 2. IMU Calibration

```cpp
class Calib {
    cv::Mat Tcb, Tbc;
    cv::Mat Cov, CovWalk;
};
```
**Explanation:** 
- `Tcb` / `Tbc`: The rigid 3D transformation (Extrinsics) between the Camera (c) and the IMU body (b). The camera and IMU are physically bolted to the drone; this matrix tells the math exactly how far apart they are.
- `Cov` / `CovWalk`: The statistical covariance (noise characteristics) of the specific IMU sensor, usually provided by the manufacturer or measured using Allan Variance.

### 3. The Preintegration Engine

```cpp
class Preintegrated {
    void IntegrateNewMeasurement(const cv::Point3f &acceleration, const cv::Point3f &angVel, const float &dt);
    void Reintegrate();
    cv::Mat GetUpdatedDeltaRotation();
    cv::Mat GetUpdatedDeltaVelocity();
    cv::Mat GetUpdatedDeltaPosition();
};
```
**Explanation:** 
- This is the most mathematically complex class in the file.
- **The Problem:** Between Camera Frame 1 (at $t=0.0s$) and Camera Frame 2 (at $t=0.033s$), the IMU has generated 7 measurements. We want to integrate those 7 measurements to figure out exactly how the drone moved. However, integration depends on the starting state (velocity and orientation). If the `Optimizer` later changes the starting orientation of Frame 1 to fix an error, we would theoretically have to re-integrate all 7 IMU measurements from scratch, which is too slow.
- **The Solution (Preintegration):** This class integrates the 7 IMU measurements in a *local, relative frame* ($\Delta R, \Delta V, \Delta P$), completely independent of the global starting pose. 
- `IntegrateNewMeasurement`: Called sequentially 7 times. It continuously accumulates the $\Delta$ rotation, velocity, and position, while also updating the statistical Information Matrix (`Info`) representing the growing uncertainty.
- **First-Order Bias Correction:** If the optimizer slightly adjusts the IMU biases, we still don't want to reintegrate from scratch. The matrices `JRg, JVg, JVa, JPg, JPa` are the Jacobians of the preintegration with respect to the biases. `GetUpdatedDeltaPosition()` uses a first-order Taylor expansion to instantly estimate what the new integration *would* be if we changed the bias, saving massive amounts of CPU time.

### 4. Lie Algebra Math Helpers

```cpp
cv::Mat ExpSO3(const float &x, const float &y, const float &z);
cv::Mat LogSO3(const cv::Mat &R);
cv::Mat RightJacobianSO3(const float &x, const float &y, const float &z);
```
**Explanation:** 
- Overloads of the Lie Algebra functions from `G2oTypes.h`, providing the same mathematical operations (mapping between 3D vectors and $3 \times 3$ Rotation Matrices) using OpenCV `cv::Mat` types instead of Eigen types for ease of use in the Tracking threads.
