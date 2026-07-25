# Documentation: `MapDrawer.h`

## High-Level Overview
The `MapDrawer.h` file defines a helper class exclusively dedicated to rendering the 3D components of the SLAM system.
While `FrameDrawer.h` handles the 2D overlays on top of the live video feed, `MapDrawer.h` uses OpenGL (via the Pangolin library) to draw the 3D `MapPoint` point cloud, the camera trajectory (the `KeyFrame`s), and the essential graph connecting them.
It isolates the raw rendering logic away from the `Viewer` thread, pulling the required data from the active `Atlas` while adhering to strict thread-safety rules to avoid reading the Map while it is being actively optimized by Bundle Adjustment.

**Primary Dependencies:**
- `Pangolin` (An OpenGL wrapper for 3D visualization).
- `Atlas.h`, `MapPoint.h`, `KeyFrame.h` (The data it needs to draw).

---

## Block-by-Block Breakdown

### 1. Rendering Functions

```cpp
void DrawMapPoints();
void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph);
void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
```
**Explanation:** 
- `DrawMapPoints`: Iterates through all MapPoints in the active map and draws them as 3D GL points. Points actively observed by the current frame are often colored red, while older, stable points are colored black.
- `DrawKeyFrames`: Draws the camera trajectory. It draws small blue pyramids representing the physical cameras. 
  - `bDrawGraph`: If true, it draws green lines connecting KeyFrames that share physical observations (the Covisibility Graph).
  - `bDrawInertialGraph`: If true, it draws lines indicating the physical path the IMU integrated between frames.
- `DrawCurrentCamera`: Draws a larger green pyramid representing the live pose of the drone at this exact millisecond.

### 2. Camera Matrix Conversion

```cpp
void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw);
```
**Explanation:** 
- OpenCV stores transformations as $4 \times 4$ matrices composed of `float` or `double` using row-major memory order.
- OpenGL requires $4 \times 4$ matrices in column-major order to render the camera viewport correctly.
- This function fetches the latest camera pose `Tcw` from the `Tracking` thread, transposes it, formats it for OpenGL, and passes it to Pangolin so the user's virtual 3D camera follows the physical drone.

### 3. Display Configuration

```cpp
float mKeyFrameSize;
float mKeyFrameLineWidth;
float mPointSize;
float mCameraSize;
```
**Explanation:** 
- The visual styling for the 3D rendering (e.g., how thick the lines are, how big the camera pyramids are). These parameters are loaded from the YAML configuration file on startup via `ParseViewerParamFile`.

```cpp
float mfFrameColors[6][3] = {{0.0f, 0.0f, 1.0f}, ...};
```
**Explanation:** 
- An array of RGB colors. In `SP_SLAM3` (via ORB-SLAM3's Atlas system), you can have multiple disjoint maps if tracking was lost and restarted. `MapDrawer` uses these different colors to visually distinguish Map 1 from Map 2 in the 3D viewer.
