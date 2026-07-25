# Documentation: `Pinhole.cpp`

## High-Level Overview
The `Pinhole.cpp` file implements the `Pinhole` geometric camera model. This is the simplest and most common mathematical model for standard cameras. It maps a 3D physical point onto a 2D image plane using classical perspective projection. 
In a SLAM system, the camera model dictates how physical 3D map points are projected into the camera frame to calculate reprojection errors, which are then minimized during bundle adjustment to correct tracking drift.

**Primary Dependencies:**
- `Pinhole.h`
- OpenCV (`cv::Point2f`, `cv::Point3f`, `cv::Mat`, `cv::KeyPoint`)
- Eigen (`Eigen::Vector2d`, `Eigen::Vector3d`, `Eigen::Matrix`)

---

## Block-by-Block Breakdown

### 1. 3D to 2D Projection Functions

```cpp
cv::Point2f Pinhole::project(const cv::Point3f &p3D) {
    return cv::Point2f(mvParameters[0] * p3D.x / p3D.z + mvParameters[2],
                       mvParameters[1] * p3D.y / p3D.z + mvParameters[3]);
}
```
**Explanation:** 
- Converts a 3D point (`p3D`) in the camera's local coordinate frame into 2D pixel coordinates on the image plane.
- **Mathematics (Perspective Projection):** 
  - `x' = fx * (X / Z) + cx`
  - `y' = fy * (Y / Z) + cy`
  - `mvParameters[0]` and `mvParameters[1]` are the focal lengths `fx` and `fy`.
  - `mvParameters[2]` and `mvParameters[3]` are the principal point offsets `cx` and `cy` (the optical center of the image).
  - The function normalizes the physical X and Y coordinates by dividing by depth `Z`, scaling them by focal length, and shifting them by the principal point.
- Overloaded methods exist for `cv::Mat` and `Eigen::Vector3d` inputs, performing the exact same calculation.

### 2. 2D to 3D Unprojection (Ray Casting)

```cpp
cv::Point3f Pinhole::unproject(const cv::Point2f &p2D) {
    return cv::Point3f((p2D.x - mvParameters[2]) / mvParameters[0], 
                       (p2D.y - mvParameters[3]) / mvParameters[1],
                       1.f);
}
```
**Explanation:** 
- Converts a 2D pixel coordinate back into a 3D ray extending outward from the camera optical center.
- **Algorithm:** Reverses the perspective projection formula. 
  - Subtracts the principal point (`cx`, `cy`) and divides by the focal lengths (`fx`, `fy`).
  - Sets the `Z` depth to `1.0f` to create a *normalized unit plane* vector. Real-world 3D points lie somewhere along this scaled ray.

### 3. Jacobians for Bundle Adjustment

```cpp
cv::Mat Pinhole::projectJac(const cv::Point3f &p3D) {
    cv::Mat Jac(2, 3, CV_32F);
    Jac.at<float>(0, 0) = mvParameters[0] / p3D.z;
    Jac.at<float>(0, 1) = 0.f;
    Jac.at<float>(0, 2) = -mvParameters[0] * p3D.x / (p3D.z * p3D.z);
    Jac.at<float>(1, 0) = 0.f;
    Jac.at<float>(1, 1) = mvParameters[1] / p3D.z;
    Jac.at<float>(1, 2) = -mvParameters[1] * p3D.y / (p3D.z * p3D.z);
    return Jac;
}
```
**Explanation:** 
- Computes the 2x3 Jacobian matrix of the pinhole projection function with respect to the 3D point coordinates (X, Y, Z).
- **Why is this necessary?** During Bundle Adjustment (g2o optimization), the SLAM engine needs to know how a tiny change in a 3D point's location affects its 2D pixel projection to minimize the total reprojection error.
- **Calculus:** This is the analytical partial derivative matrix of the projection function `[u(x,y,z), v(x,y,z)]`. For example, the derivative of `u` with respect to `x` is `fx / Z` (stored at `Jac(0,0)`). The derivative of `u` with respect to `z` requires the quotient rule, resulting in `-fx * X / Z^2` (stored at `Jac(0,2)`).

```cpp
cv::Mat Pinhole::unprojectJac(const cv::Point2f &p2D) {
    cv::Mat Jac(3, 2, CV_32F);
    Jac.at<float>(0, 0) = 1 / mvParameters[0];
    // ...
```
**Explanation:** 
- Computes the 3x2 Jacobian of the unprojection function. Simply the inverse of the focal lengths.

### 4. Epipolar Geometry Validation

```cpp
bool Pinhole::epipolarConstrain(GeometricCamera* pCamera2, const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &R12, const cv::Mat &t12, const float sigmaLevel, const float unc) {
    //Compute Fundamental Matrix
    cv::Mat t12x = SkewSymmetricMatrix(t12);
    cv::Mat K1 = this->toK();
    cv::Mat K2 = pCamera2->toK();
    cv::Mat F12 = K1.t().inv()*t12x*R12*K2.inv();

    // Epipolar line in second image l = x1'F12 = [a b c]
    // ...
    const float dsqr = num*num/den;
    return dsqr<3.84*unc;
}
```
**Explanation:** 
- Determines if two matched 2D points (from two different camera frames) obey epipolar geometry constraints before attempting 3D triangulation. This filters out false feature matches.
- **Algorithm (Epipolar Constraint):** 
  1. Computes the **Fundamental Matrix** `F12` using the relative rotation `R12`, relative translation `t12` (converted to a skew-symmetric matrix `t12x`), and both cameras' intrinsic `K` matrices. 
  2. The Fundamental Matrix constraint dictates that `p2^T * F12 * p1 = 0`. Geometrically, the point `p1` in the first camera creates an "epipolar line" in the second camera's image (`l = F12 * p1`). The matched point `p2` MUST lie on or very near this line.
  3. It calculates the perpendicular squared distance `dsqr` from the matched point `kp2` to the epipolar line `[a b c]`.
  4. Returns true if the distance is less than the statistical Chi-Square threshold (`3.84` for 1 degree of freedom), meaning the match is geometrically valid.

### 5. Utility Functions

```cpp
cv::Mat Pinhole::SkewSymmetricMatrix(const cv::Mat &v)
```
**Explanation:** 
- Converts a 3x1 translation vector `v` into a 3x3 skew-symmetric cross-product matrix. This is a fundamental operation in epipolar geometry used to represent the cross product as matrix multiplication (`v x a = [v]_x * a`).
