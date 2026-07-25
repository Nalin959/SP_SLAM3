# Documentation: `GeometricCamera.h`

## High-Level Overview
The `GeometricCamera.h` file defines an abstract base class (`GeometricCamera`) that establishes the interface for all camera models used within the SLAM system. 
In modern visual SLAM, assuming a simple pinhole camera is often insufficient, especially when using ultra-wide or fisheye lenses common on drones and robotics. ORB-SLAM3 introduced this abstraction layer so the tracking and mapping logic can be entirely decoupled from the specific physics and distortion models of the lenses.

This abstract class demands that any derived class (e.g., `Pinhole`, `KannalaBrandt8`) implements core mathematical functions like projecting a 3D point into 2D pixels, unprojecting a 2D pixel into a 3D ray, and calculating the mathematical Jacobians for these operations (which are strictly required by the g2o optimizer).

**Primary Dependencies:**
- OpenCV (`cv::Point2f`, `cv::Point3f`, `cv::Mat`)
- Eigen (`Eigen::Vector2d`, `Eigen::Vector3d`, `Eigen::Matrix`)
- Boost Serialization (For saving camera intrinsics to disk alongside the map).

---

## Block-by-Block Breakdown

### 1. Boost Serialization Interface

```cpp
class GeometricCamera {
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
```
**Explanation:** 
- To save the global `Atlas` state, we must serialize the cameras.
- This base serialization function saves the global ID of the camera (`mnId`), the type enum (`mnType`: Pinhole or Fisheye), and the raw underlying mathematical parameters (`mvParameters` like $f_x, f_y, c_x, c_y$, etc.).

### 2. The Abstract Projection Interface

```cpp
virtual cv::Point2f project(const cv::Point3f &p3D) = 0;
virtual cv::Point2f project(const cv::Mat& m3D) = 0;
virtual Eigen::Vector2d project(const Eigen::Vector3d & v3D) = 0;
virtual cv::Mat projectMat(const cv::Point3f& p3D) = 0;
```
**Explanation:** 
- The `= 0` syntax in C++ makes these **pure virtual** functions. This enforces that `GeometricCamera` cannot be instantiated directly; it must be subclassed.
- **Projection:** These functions take a physical 3D point in the camera's local coordinate frame $(X_c, Y_c, Z_c)$ and apply the specific lens distortion model to return the 2D $(u, v)$ pixel coordinates where that point will appear on the image sensor. It provides multiple overloads for convenience (OpenCV vs. Eigen).

### 3. The Abstract Unprojection Interface

```cpp
virtual cv::Point3f unproject(const cv::Point2f &p2D) = 0;
virtual cv::Mat unprojectMat(const cv::Point2f &p2D) = 0;
```
**Explanation:** 
- **Unprojection:** This is the inverse of projection. Given a 2D pixel coordinate $(u,v)$, it returns a normalized 3D ray $(X, Y, 1.0)$ pointing out from the camera center into the world. It strips away the specific lens distortion mathematically.

### 4. Jacobians for Bundle Adjustment

```cpp
virtual cv::Mat projectJac(const cv::Point3f &p3D) = 0;
virtual Eigen::Matrix<double,2,3> projectJac(const Eigen::Vector3d& v3D) = 0;
virtual cv::Mat unprojectJac(const cv::Point2f &p2D) = 0;
```
**Explanation:** 
- When the `Optimizer` runs Bundle Adjustment (g2o), it needs to know exactly how a tiny change in a 3D MapPoint's position will affect its projected 2D pixel location. This derivative is the Jacobian matrix.
- Because different lenses distort light differently, the mathematical derivative of the projection function is unique to each camera model. The base class forces derived classes to provide these exact analytical derivatives.

### 5. Multi-View Geometry Contracts

```cpp
virtual bool ReconstructWithTwoViews(...) = 0;
virtual bool epipolarConstrain(...) = 0;
virtual bool matchAndtriangulate(...) = 0;
```
**Explanation:** 
- Multi-view geometry (like triangulating a 3D point from two 2D images, or searching along an epipolar line) is highly dependent on the lens.
- Standard Epipolar geometry assumes straight lines, which breaks down entirely in Fisheye lenses (where straight lines curve).
- By pushing these functions into the `GeometricCamera` interface, the SLAM system can seamlessly initialize a map or triangulate points whether the drone has standard webcams or extreme 180-degree fisheye lenses.
