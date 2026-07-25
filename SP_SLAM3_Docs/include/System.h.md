# Documentation: `System.h`

## High-Level Overview
The `System.h` file acts as the primary API and absolute root of the entire `SP_SLAM3` application. 
When a user writes a custom C++ application (e.g., a ROS node, an Android app, or a drone autopilot) and wants to use this SLAM engine, they include this single header file.
The `System` class is responsible for booting up the entire architecture, allocating the neural networks, launching the background threads (`LocalMapping`, `LoopClosing`, `Viewer`), and exposing the core `Track(...)` functions that the user's camera feed will pump images into.

**Primary Dependencies:**
- `Tracking.h`, `LocalMapping.h`, `LoopClosing.h`, `Viewer.h` (The primary threads).
- `Atlas.h` (The root map database).

---

## Block-by-Block Breakdown

### 1. Initialization and Thread Launching

```cpp
System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, ...);
std::thread* mptLocalMapping;
std::thread* mptLoopClosing;
std::thread* mptViewer;
```
**Explanation:** 
- The constructor is massive. It takes the path to the DBoW3 vocabulary file and the YAML settings file (which contains camera intrinsics and neural network paths).
- It allocates the `Atlas` and `KeyFrameDatabase`.
- It instantiates the `Tracking`, `LocalMapping`, and `LoopClosing` objects, passing pointers between them so they can communicate.
- Finally, it launches `LocalMapping`, `LoopClosing`, and the `Viewer` in their own `std::thread`s. The `System` object itself (and the `Tracking` object it owns) lives in the main application thread.

### 2. The Main Execution API (Tracking)

```cpp
cv::Mat TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, ...);
cv::Mat TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, ...);
cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point>& vImuMeas, ...);
```
**Explanation:** 
- The user's application calls one of these functions in a fast `while(true)` loop as camera frames arrive.
- They accept raw OpenCV images and (optionally) a vector of IMU measurements that occurred since the last frame.
- They pass this data synchronously to the `Tracking` thread and return a `cv::Mat` representing the real-time $4 \times 4$ camera pose (Rotation and Translation relative to the starting point of the flight). If tracking is lost, it returns an empty matrix.

### 3. Trajectory Exporting

```cpp
void SaveTrajectoryTUM(const string &filename);
void SaveKeyFrameTrajectoryTUM(const string &filename);
void SaveTrajectoryKITTI(const string &filename);
```
**Explanation:** 
- SLAM algorithms are often benchmarked using standardized datasets (TUM, KITTI, EuRoC). 
- Once a flight is complete, the user calls `Shutdown()` to kill the background threads cleanly, and then calls one of these functions to dump the computed trajectory to a text file in a standardized format so it can be compared against Ground Truth.

### 4. Mode Switching

```cpp
void ActivateLocalizationMode();
void DeactivateLocalizationMode();
```
**Explanation:** 
- In standard SLAM, the drone builds the map and tracks against it simultaneously.
- However, if you fly a drone through a building once to build a perfect map, you can save that map. The next day, you can load the map and call `ActivateLocalizationMode()`. 
- This disables the `LocalMapping` thread entirely. The drone will purely track its position against the existing map without modifying it, triangulating new points, or consuming extra CPU.
