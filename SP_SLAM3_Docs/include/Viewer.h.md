# Documentation: `Viewer.h`

## High-Level Overview
The `Viewer.h` file defines the 3D visualization and UI thread for the SLAM system.
While the SLAM algorithm (Tracking, LocalMapping, LoopClosing) runs entirely on math and matrices in the background, developers and users need to *see* what the drone is doing to debug it or understand its trajectory. 
The `Viewer` class spawns a dedicated UI thread that uses the **Pangolin** OpenGL library. It continuously pulls the latest data from the `FrameDrawer` (for 2D camera images) and the `MapDrawer` (for the 3D point cloud and camera trajectory) and renders it to a live interactive window at 30-60 FPS.

**Primary Dependencies:**
- `Pangolin` (The underlying OpenGL windowing and rendering library, though hidden in the `.cpp` file).
- `FrameDrawer.h` (For 2D image overlays).
- `MapDrawer.h` (For 3D map rendering).
- `System.h` and `Tracking.h` (To control execution state).

---

## Block-by-Block Breakdown

### 1. Initialization and Execution

```cpp
Viewer(System* pSystem, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Tracking *pTracking, const string &strSettingPath);
void Run();
```
**Explanation:** 
- The constructor takes pointers to the drawing helper classes and parses the YAML settings file (via `ParseViewerParamFile`) to configure the initial camera viewpoint (e.g., placing the virtual camera slightly behind and above the drone, looking down at an angle).
- `Run()`: The infinite loop executed by the `mptViewer` thread. Inside this loop, it clears the OpenGL buffers, calls the drawing functions from `MapDrawer` and `FrameDrawer`, and swaps the buffers to the screen. It also sleeps (`mT`) to ensure it doesn't run faster than the requested framerate, saving CPU/GPU cycles.

### 2. Thread-Safe State Control

```cpp
void RequestFinish();
void RequestStop();
bool isFinished();
bool isStopped();
```
**Explanation:** 
- Because the `Viewer` is a blocking infinite loop running in its own thread, it cannot simply be killed or paused without causing OpenGL context crashes or memory leaks.
- `RequestStop()`: If the user clicks the "Pause" button in the Pangolin UI, the Viewer politely asks the `Tracking` thread to pause SLAM execution.
- `RequestFinish()`: When the user closes the application (or the dataset finishes), the main `System` thread calls this. It sets a boolean flag protected by a `std::mutex`. The `Run()` loop checks this flag every frame and exits cleanly if it is set to true.

### 3. Step-By-Step Debugging

```cpp
bool isStepByStep();
void SetTrackingPause();
```
**Explanation:** 
- `SetTrackingPause`: A powerful debugging tool. If enabled, the SLAM system will process exactly one camera frame and then freeze. The user can inspect the 3D map, look at the extracted SuperPoint features, and then click a "Step" button in the UI to process exactly one more frame. This is crucial for debugging why a specific frame caused tracking to fail.

### 4. Rendering Configuration

```cpp
double mT;
float mImageWidth, mImageHeight;
float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;
```
**Explanation:** 
- `mT`: The target frame time in milliseconds (e.g., $33.3$ ms for 30 FPS).
- `mViewpointX, Y, Z, F`: These define the $SE(3)$ pose of the *virtual* camera looking at the 3D map, along with its focal length (`F`). By tweaking these in the YAML file, the user can change whether the UI starts in a top-down "God view" or a third-person "Follow view" directly behind the drone.
