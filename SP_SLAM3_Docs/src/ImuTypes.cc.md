# Documentation: `ImuTypes.cc`

## High-Level Overview
The `ImuTypes.cc` file provides the foundational mathematics and data structures required to integrate Inertial Measurement Unit (IMU) data into the SLAM pipeline. 
Cameras operate at ~30 Hz, while IMUs operate at ~200 Hz. Processing every IMU measurement as an independent optimization node would cause the graph to explode in size. Therefore, this file implements **IMU Preintegration** (based on Forster et al., 2015), which mathematically compresses hundreds of high-frequency IMU readings between two camera frames into a single, unified "Preintegrated Measurement" edge. 
It also implements the Lie Algebra ($SO(3)$) functions necessary to properly integrate 3D rotations without hitting Gimbal Lock or breaking orthogonality.

**Primary Dependencies:**
- `ImuTypes.h`
- OpenCV (`cv::Mat`, `cv::SVD`)
- Eigen (for $SO(3)$ mappings)

---

## Block-by-Block Breakdown

### 1. Lie Algebra Math (SO(3) Manifold)

```cpp
cv::Mat ExpSO3(const float &x, const float &y, const float &z)
{
    // ...
    cv::Mat W = (cv::Mat_<float>(3,3) << 0, -z, y, z, 0, -x, -y, x, 0);
    if(d<eps)
        return (I + W + 0.5f*W*W);
    else
        return (I + W*sin(d)/d + W*W*(1.0f-cos(d))/d2);
}
```
**Explanation:** 
- **The Exponential Map:** Converts a 3D rotation vector (the Lie Algebra $\mathfrak{so}(3)$, representing the axis-angle of rotation) into a 3x3 valid rotation matrix (the Lie Group $SO(3)$).
- Uses the **Rodrigues' Rotation Formula**. 
- It explicitly handles the mathematical singularity near zero (`d < eps`) by using a Taylor series approximation (`I + W + 0.5*W^2`) to avoid dividing by zero.

```cpp
cv::Mat LogSO3(const cv::Mat &R)
```
**Explanation:** 
- **The Logarithmic Map:** The exact inverse of `ExpSO3`. Takes a 3x3 Rotation matrix and maps it back to a 3D tangent vector. Used heavily when computing the rotational error during optimization.

```cpp
cv::Mat RightJacobianSO3(const float &x, const float &y, const float &z)
cv::Mat InverseRightJacobianSO3(...)
```
**Explanation:** 
- Unlike standard Euclidean space, where $X = X_0 + \Delta X$, updating a rotation matrix requires matrix multiplication on the manifold ($R = R_0 \cdot \text{Exp}(\Delta \theta)$). 
- The Right Jacobian of $SO(3)$ maps how a tiny change in the tangent space ($\Delta \theta$) propagates through the non-linear exponential map. These matrices are required by `g2o` to correctly linearize the optimization problem.

### 2. IMU Preintegration Core

```cpp
void Preintegrated::IntegrateNewMeasurement(const cv::Point3f &acceleration, const cv::Point3f &angVel, const float &dt)
```
**Explanation:** 
- This is the heartbeat of the IMU pipeline. It is called every time a new IMU sample arrives (e.g., 200 times a second).
- **Algorithm:**
  1. It subtracts the current estimated biases (`b.bax`, `b.bwx`) from the raw accelerometer and gyroscope readings.
  2. It updates the relative position (`dP`) and velocity (`dV`) using basic kinematics ($\Delta V = a \cdot dt$, $\Delta P = V \cdot dt + 0.5 \cdot a \cdot dt^2$), rotating the acceleration vector by the current preintegrated rotation `dR`.
  3. It computes massive 9x9 Jacobian blocks (`A`, `B`) and updates the `15x15` Covariance matrix `C`. This tracks the accumulated uncertainty (noise + random walk) over time.
  4. Finally, it updates the relative rotation `dR` using `ExpSO3`, and tracks how much the rotation would change if the gyro bias were slightly adjusted (`JRg`).

### 3. Bias Correction and Re-integration

```cpp
void Preintegrated::Reintegrate()
{
    // ...
    Initialize(bu);
    for(size_t i=0;i<aux.size();i++)
        IntegrateNewMeasurement(aux[i].a,aux[i].w,aux[i].t);
}
```
**Explanation:** 
- If the global optimization significantly changes the estimated IMU biases (`bu`), the preintegrated measurement mathematically breaks.
- This function throws away the current integration, takes the raw measurements cached in `mvMeasurements`, and perfectly recalculates `dP, dV, dR` from scratch using the new biases.

```cpp
cv::Mat Preintegrated::GetUpdatedDeltaRotation()
cv::Mat Preintegrated::GetUpdatedDeltaVelocity()
cv::Mat Preintegrated::GetUpdatedDeltaPosition()
```
**Explanation:** 
- Reintegrating from scratch (above) is computationally heavy.
- During the middle of a `g2o` optimization step, when the biases are changing slightly every iteration, these functions use a **First-Order Taylor Approximation** to instantly guess the new preintegrated values without running the full loop. 
- Example: `dV_new = dV_old + JVg * delta_gyro_bias + JVa * delta_accel_bias`.

### 4. Calibration Structs

```cpp
void Calib::Set(const cv::Mat &Tbc_, const float &ng, const float &na, const float &ngw, const float &naw)
```
**Explanation:** 
- Stores the hardware-specific intrinsic calibration for the IMU.
- `Tbc`: The rigid geometric transformation from the IMU Body frame to the Camera frame.
- `ng`, `na`: The white noise density of the gyroscope and accelerometer (from the datasheet).
- `ngw`, `naw`: The random walk (bias drift over time) of the sensors.
- It builds the base 6x6 Covariance matrices used to inject noise into the preintegration `IntegrateNewMeasurement` step.
