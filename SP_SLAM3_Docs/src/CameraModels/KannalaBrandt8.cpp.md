# Documentation: `KannalaBrandt8.cpp`

## High-Level Overview
The `KannalaBrandt8.cpp` file implements the `KannalaBrandt8` camera model class. This is a specific geometric camera model used for wide-angle and fisheye lenses, developed by Juho Kannala and Sami S. Brandt. 
Unlike the standard Pinhole camera model which fails or severely distorts at the edges of fisheye lenses, the Kannala-Brandt model uses a polynomial projection function (up to the 9th degree of theta) to accurately map 3D points onto the 2D image plane of a highly distorted lens.

**Primary Dependencies:**
- `KannalaBrandt8.h`
- OpenCV (`cv::Point2f`, `cv::Point3f`, `cv::Mat`, `cv::fisheye`)
- Eigen (`Eigen::Vector2d`, `Eigen::Vector3d`, `Eigen::Matrix`)

---

## Block-by-Block Breakdown

### 1. 3D to 2D Projection Functions

```cpp
cv::Point2f KannalaBrandt8::project(const cv::Point3f &p3D) {
    const float x2_plus_y2 = p3D.x * p3D.x + p3D.y * p3D.y;
    const float theta = atan2f(sqrtf(x2_plus_y2), p3D.z);
    const float psi = atan2f(p3D.y, p3D.x);

    const float theta2 = theta * theta;
    const float theta3 = theta * theta2;
    // ... higher powers of theta ...
    const float r = theta + mvParameters[4] * theta3 + mvParameters[5] * theta5
                    + mvParameters[6] * theta7 + mvParameters[7] * theta9;

    return cv::Point2f(mvParameters[0] * r * cos(psi) + mvParameters[2],
                       mvParameters[1] * r * sin(psi) + mvParameters[3]);
}
```
**Explanation:** 
- Converts a 3D point in the camera coordinate frame (`p3D`) into a 2D pixel coordinate on the image plane.
- **Mathematics:** It calculates the incident angle `theta` relative to the optical axis (Z-axis) and the azimuthal angle `psi`.
- The radial distance `r` on the image plane is computed using a 9th-order polynomial of `theta`. The coefficients `mvParameters[4]` through `mvParameters[7]` represent the `k1`, `k2`, `k3`, and `k4` distortion parameters of the fisheye lens.
- Finally, it uses the focal lengths (`mvParameters[0]`, `mvParameters[1]`) and the principal point offsets (`mvParameters[2]`, `mvParameters[3]`) to map the distorted coordinates to exact pixel coordinates.
- Overloaded methods exist for `cv::Mat` and `Eigen::Vector3d` inputs, doing the exact same mathematical operation.

### 2. 2D to 3D Unprojection (Ray Casting)

```cpp
cv::Point3f KannalaBrandt8::unproject(const cv::Point2f &p2D) {
    //Use Newthon method to solve for theta with good precision (err ~ e-6)
    cv::Point2f pw((p2D.x - mvParameters[2]) / mvParameters[0], (p2D.y - mvParameters[3]) / mvParameters[1]);
    float scale = 1.f;
    float theta_d = sqrtf(pw.x * pw.x + pw.y * pw.y);
    theta_d = fminf(fmaxf(-CV_PI / 2.f, theta_d), CV_PI / 2.f);
    // ...
```
**Explanation:** 
- Converts a 2D pixel coordinate back into a 3D unit ray (or a scaled 3D point) pointing outward from the camera.
- **Algorithm:** Because the 9th-order polynomial projection function cannot be trivially inverted analytically, this function uses the **Newton-Raphson method** (an iterative root-finding algorithm) to solve for the original incident angle `theta`.
- It iterates up to 10 times, updating `theta` by subtracting `theta_fix` (the function value divided by its derivative), breaking early if the precision `1e-6` is met.
- **Performance Optimization:** Instead of doing a heavy generic optimization, the derivative (`1 + 3*k0*theta^2 + ...`) is hardcoded mathematically for extremely fast execution during the SLAM pipeline.

### 3. Jacobians for Bundle Adjustment

```cpp
cv::Mat KannalaBrandt8::projectJac(const cv::Point3f &p3D) {
    // ... setup theta and its powers ...
    float f = theta + theta3 * mvParameters[4] + theta5 * mvParameters[5] + ...
    float fd = 1 + 3 * mvParameters[4] * theta2 + 5 * mvParameters[5] * theta4 + ...

    cv::Mat Jac(2, 3, CV_32F);
    Jac.at<float>(0, 0) = mvParameters[0] * (fd * p3D.z * x2 / (r2 * (r2 + z2)) + f * y2 / r3);
    // ... remaining partial derivatives ...
    return Jac.clone();
}
```
**Explanation:** 
- Computes the 2x3 Jacobian matrix of the projection function with respect to the 3D point coordinates (X, Y, Z).
- **Why is this necessary?** During Bundle Adjustment (a non-linear least squares optimization used to refine the map and camera poses), the optimizer (g2o) requires the partial derivatives of the error function to calculate the gradient and update the parameters.
- Providing an analytical Jacobian (exact mathematical derivative) is much faster and more accurate than using numerical finite differences. 
- Overloaded for `Eigen::Vector3d`.

### 4. Epipolar Geometry & Triangulation

```cpp
bool KannalaBrandt8::matchAndtriangulate(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, GeometricCamera* pOther,
                                         cv::Mat& Tcw1, cv::Mat& Tcw2,
                                         const float sigmaLevel1, const float sigmaLevel2,
                                         cv::Mat& x3Dtriangulated)
```
**Explanation:** 
- Given two matched 2D features (`kp1` from this camera, `kp2` from `pOther`), this function determines if they represent a valid 3D point in the real world.
- **Algorithm (Epipolar Check):** 
  1. It unprojects both 2D points into 3D unit rays (`ray1c`, `ray2c`).
  2. It rotates the rays into the global world coordinate frame (`Rwc1`, `Rwc2`).
  3. **Parallax Check:** It calculates the dot product between the two rays. If `cosParallaxRays > 0.9998`, the angle between the rays is too small (nearly parallel). Triangulating parallel rays leads to massive depth uncertainty, so the match is immediately rejected.
  4. If parallax is sufficient, it calls `Triangulate()` to find the 3D intersection point using Direct Linear Transformation (DLT).
  5. **Cheirality Check:** It checks if the triangulated 3D point is strictly *in front* of both cameras (`z > 0`). Points behind the camera are physically impossible and indicate a false match.
  6. **Reprojection Error:** The 3D point is projected *back* into the 2D image planes. The squared pixel distance between the original feature and the reprojected point must fall under a Chi-Square threshold (`5.991 * sigmaLevel`).
- Returns `true` and populates `x3Dtriangulated` if the point survives all geometric checks.

```cpp
void KannalaBrandt8::Triangulate(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Mat &Tcw1, const cv::Mat &Tcw2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    A.row(0) = p1.x*Tcw1.row(2)-Tcw1.row(0);
    // ...
    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}
```
**Explanation:** 
- Performs Linear Triangulation using Singular Value Decomposition (SVD).
- It constructs the classical DLT `A` matrix using the known camera poses (`Tcw1`, `Tcw2`) and the 2D ray coordinates (`p1`, `p2`).
- The optimal 3D point in homogeneous coordinates is the right singular vector corresponding to the smallest singular value (the last row of `vt`).
- Finally, it normalizes the homogeneous point by dividing by the 4th coordinate (`x3D.at<float>(3)`) to get standard 3D Euclidean coordinates.
