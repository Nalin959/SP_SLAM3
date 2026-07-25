# Documentation: `System.cc`

## High-Level Overview
The `System.cc` file is the master orchestrator of the entire SLAM application. It serves as the primary API for external applications (like ROS nodes, Pangolin viewers, or command-line wrappers) to interact with the SLAM pipeline.
When the `System` object is instantiated, it parses the configuration files, allocates memory for the global map (`Atlas`), loads the deep learning models (SuperPoint, LightGlue, NetVLAD), and spins up the three core concurrent threads that define the ORB-SLAM architecture: **Tracking**, **Local Mapping**, and **Loop Closing**.
After initialization, the user calls `TrackMonocular()`, `TrackStereo()`, or `TrackRGBD()` in a loop, feeding images into the system frame by frame.

**Primary Dependencies:**
- `System.h`
- `Tracking.h`, `LocalMapping.h`, `LoopClosing.h` (The big three SLAM threads).
- `LightGlue.h`, `PlaceRecognition.h` (Deep Learning models).
- `Atlas.h` (The multi-map global state).
- `Pangolin` (For the 3D Viewer thread).

---

## Block-by-Block Breakdown

### 1. System Initialization (The Constructor)

```cpp
System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, ...)
```
**Explanation:** 
- **Configuration Parsing:** It immediately opens the user-provided `.yaml` settings file using `cv::FileStorage` to read camera intrinsics, FPS, and algorithmic parameters.
- **Vocabulary Loading:** It loads the massive Bag-of-Words vocabulary file (`.txt` or `.bin`), which takes a few seconds but is critical for Loop Closing and Relocalization.
- **Map Allocation:** It initializes the `Atlas` (the container for one or multiple independent `Map`s) and the `KeyFrameDatabase` (the inverted index used for searching).
- **Thread Initialization:**
  - `mpTracker`: Runs synchronously on the calling thread. It is instantiated but not started as a `std::thread`.
  - `mpLocalMapper`: Instantiated and launched as a background `std::thread`.
  - `mpLoopCloser`: Instantiated and launched as a background `std::thread`.
  - `mpViewer`: If requested, launched as a background `std::thread`.
- **Deep Learning Loading:**
  - It probes the settings file for `LightGlue.model_path` and `PlaceRecognition.model_path`.
  - If found, it instantiates the LibTorch models on the GPU and injects the pointers into the Tracker and LoopCloser. This is a massive modification over original ORB-SLAM3.
- **Thread Linkage:** Finally, it passes pointers of every thread to every other thread so they can communicate (e.g., Local Mapping needs to tell Tracking to pause when it runs Bundle Adjustment).

### 2. The Tracking Entry Points

```cpp
cv::Mat System::TrackMonocular(...)
cv::Mat System::TrackStereo(...)
cv::Mat System::TrackRGBD(...)
```
**Explanation:** 
- These are the main loop functions called by the user on every new camera frame.
- **State Management:** Before processing the frame, it locks the `mMutexMode` and `mMutexReset` to check if the user has requested a system reset or a mode switch (e.g., turning off Mapping to run in "Localization Only" mode).
- **IMU Injection:** If the system is running Visual-Inertial SLAM, it pushes the high-frequency IMU measurements (accelerometer and gyroscope data) into the tracker's buffer (`mpTracker->GrabImuData()`) before passing the image.
- **Execution:** It calls the respective `mpTracker->GrabImage...()` function, which performs the actual tracking math.
- **Return Value:** It returns the calculated $4 \times 4$ camera pose matrix `Tcw` (Transformation from World to Camera). If tracking is lost, it returns an empty matrix.

### 3. System State Controls

```cpp
void System::ActivateLocalizationMode()
void System::Reset()
void System::Shutdown()
```
**Explanation:** 
- **Localization Mode:** Kills the Local Mapping thread permanently. The system will track against the existing map but will not add new KeyFrames or MapPoints. Used for deploying pre-mapped environments.
- **Reset:** Sends a flag to the Tracker to clear the Atlas and restart tracking from scratch.
- **Shutdown:** This is crucial. When the application closes, it cannot just `exit()`. `Shutdown()` politely requests the Local Mapping and Loop Closing threads to finish their current work (e.g., finishing an active Global Bundle Adjustment), joins them, and then safely shuts down the Pangolin OpenGL viewer.

### 4. Trajectory Serialization

```cpp
void System::SaveTrajectoryTUM(...)
void System::SaveTrajectoryKITTI(...)
void System::SaveTrajectoryEuRoC(...)
```
**Explanation:** 
- Researchers and developers need to export the SLAM trajectory to evaluate accuracy against ground truth using tools like `evo`.
- Because SLAM threads run asynchronously, the raw trajectory returned by `Track()` might drift. To get the highest accuracy, these functions traverse the entire `Atlas`.
- They take the optimized KeyFrame poses (refined by Bundle Adjustment) and calculate the relative pose of every intermediate `Frame` to its parent KeyFrame.
- The output formats differ slightly:
  - **TUM:** `timestamp tx ty tz qx qy qz qw`
  - **KITTI:** $3 \times 4$ projection matrix flattened to a 12-element row.
  - **EuRoC:** Similar to TUM, but accounts for IMU sensor extrinsics (`Tbc`) if Visual-Inertial SLAM was used.
