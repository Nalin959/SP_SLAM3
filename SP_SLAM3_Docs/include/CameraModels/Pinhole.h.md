# Documentation: `Pinhole.h`

## High-Level Overview
The `Pinhole.h` file declares the implementation of the `GeometricCamera` interface for the classic **Pinhole Camera Model**.
This model assumes that light passes through a single infinitesimal point (the pinhole) and projects onto a flat image plane. It is the standard, simplest model used in computer vision for lenses with a relatively narrow field of view (e.g., standard smartphone cameras, webcams, or narrow drone cameras).
Unlike the `KannalaBrandt8` model, the Pinhole model defined here assumes that images have *already been undistorted* (rectified) by a pre-processing step. Therefore, it only requires 4 parameters: the focal lengths and the optical center.

**Primary Dependencies:**
- `GeometricCamera.h` (The base abstract class).
- `TwoViewReconstruction.h` (Used for monocular map initialization).

---

## Block-by-Block Breakdown

### 1. Class Definition and Serialization

```cpp
class Pinhole : public GeometricCamera {
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
```
**Explanation:** 
- Inherits from `GeometricCamera`. 
- **Serialization:** It only needs to serialize the base class object (which holds the `mvParameters` array containing the 4 pinhole parameters).

### 2. Constructors and Parameters

```cpp
Pinhole()
Pinhole(const std::vector<float> _vParameters)
```
**Explanation:** 
- The constructor explicitly expects an array of exactly 4 parameters (`assert(mvParameters.size() == 4);`).
- The expected order of these parameters is:
  - `[f_x, f_y, c_x, c_y]`
  - $f_x, f_y$: Focal lengths in terms of pixels.
  - $c_x, c_y$: The principal point (the exact pixel coordinate where the optical axis intersects the image plane).
- `mnType = CAM_PINHOLE`: Flags this camera to the rest of the SLAM system as a standard pinhole.

### 3. Overridden Interface Functions (The Mathematics)

```cpp
cv::Point2f project(const cv::Point3f &p3D);
cv::Point3f unproject(const cv::Point2f &p2D);
```
**Explanation:** 
- **`project`**: The math here is very simple. Given a 3D point $X_c, Y_c, Z_c$ in the camera frame, it projects it to a 2D pixel $u, v$:
  - $u = f_x \cdot (X_c / Z_c) + c_x$
  - $v = f_y \cdot (Y_c / Z_c) + c_y$
- **`unproject`**: The inverse is equally simple. Given a pixel $u, v$, it returns the 3D ray $(X, Y, 1.0)$:
  - $X = (u - c_x) / f_x$
  - $Y = (v - c_y) / f_y$
- Unlike the Kannala-Brandt model, these equations are perfectly linear and do not require iterative numerical solvers.

### 4. Jacobians for Bundle Adjustment

```cpp
cv::Mat projectJac(const cv::Point3f &p3D);
cv::Mat unprojectJac(const cv::Point2f &p2D);
```
**Explanation:** 
- Computes the $2 \times 3$ analytical derivative (Jacobian) of the projection function with respect to the 3D point $(X, Y, Z)$.
- Because the pinhole equations are simple divisions, the calculus derivatives are straightforward $\left( \frac{\partial u}{\partial X} = \frac{f_x}{Z}, \frac{\partial u}{\partial Z} = -\frac{f_x \cdot X}{Z^2}, \text{etc.} \right)$. These derivatives are fed directly into the g2o optimizer.

### 5. Standard Epipolar Geometry

```cpp
bool epipolarConstrain(GeometricCamera* pCamera2, const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, const cv::Mat& R12, const cv::Mat& t12, const float sigmaLevel, const float unc);
```
**Explanation:** 
- Computes the standard Epipolar Constraint. Given a point in image 1, it computes the Fundamental Matrix $F$ and checks if the point in image 2 lies close to the resulting epipolar line $l_2 = F x_1$.
- Because it's a pinhole camera, the epipolar search space is a perfectly straight line, unlike the complex curve of the fisheye model.
