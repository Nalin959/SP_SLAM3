# Documentation: `Viewer.cc`

## High-Level Overview
The `Viewer.cc` file is responsible for the 3D and 2D graphical visualization of the SLAM system in real-time. 
It uses the **Pangolin** library (a lightweight OpenGL windowing and UI framework popular in robotics) to render the 3D map, the trajectory, the current camera frustum, and the real-time feature extraction overlay.
Because it runs on its own isolated `std::thread`, it operates asynchronously from the Tracking and Mapping threads, meaning that even if the SLAM algorithm stutters or drops frames, the UI remains responsive and can be freely navigated by the user.

**Primary Dependencies:**
- `Viewer.h`
- `pangolin/pangolin.h` (The 3D OpenGL backend).
- `FrameDrawer.h` (Draws 2D elements onto the current OpenCV frame).
- `MapDrawer.h` (Draws 3D elements like KeyFrames and MapPoints).
- `System.h`, `Tracking.h` (For pausing/resetting the system via the UI).

---

## Block-by-Block Breakdown

### 1. Initialization and Parameter Loading

```cpp
Viewer::Viewer(System* pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking, const string &strSettingPath)
bool Viewer::ParseViewerParamFile(cv::FileStorage &fSettings)
```
**Explanation:** 
- The constructor takes pointers to the global `System` and the drawing helper classes.
- It parses the `.yaml` configuration file to determine:
  - The UI refresh rate (`Camera.fps`). The Viewer thread will intentionally sleep (`cv::waitKey(mT)`) to match this FPS to avoid consuming 100% of a CPU core on rendering.
  - The initial viewpoint of the 3D camera (`Viewer.ViewpointX`, `Y`, `Z`, and focal length `F`).

### 2. The Rendering Loop (`Run`)

```cpp
void Viewer::Run()
```
**Explanation:** 
- This function contains the infinite `while(1)` loop that runs the UI thread.
- **Pangolin Setup:** 
  - `pangolin::CreateWindowAndBind`: Opens the OS-level GUI window.
  - `glEnable(GL_DEPTH_TEST)`: Enables 3D occlusion (so objects in front hide objects behind them).
  - `pangolin::CreatePanel("menu")`: Creates the checkbox menu on the left side of the screen.
- **UI Toggles (Checkboxes):** 
  - It defines various `pangolin::Var<bool>` objects. These appear as clickable checkboxes in the UI and their boolean values update automatically.
  - Examples: `menuFollowCamera` (locks the view to the drone), `menuTopView` (switches to a bird's-eye view), `menuShowGraph` (draws the green lines connecting KeyFrames).
- **The Main Loop:**
  - `glClear`: Clears the OpenGL buffers for the new frame.
  - `mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc, Ow, Twwp)`: Safely locks the Map and extracts the drone's current pose.
  - **Camera Controls:** It evaluates the state of the checkboxes. If `menuFollowCamera` is checked, it forces the Pangolin renderer's virtual camera (`s_cam`) to update its projection matrix to follow the drone (`Twc` or `Ow`).
  - **3D Drawing:**
    - `d_cam.Activate(s_cam)`: Sets the OpenGL context.
    - `mpMapDrawer->DrawCurrentCamera`, `DrawKeyFrames`, `DrawMapPoints`: Dispatches the actual OpenGL `glBegin` / `glEnd` drawing commands to render the 3D geometry.
  - **2D Drawing (OpenCV):**
    - It asks the `FrameDrawer` to draw the raw image with green/red feature squares overlayed (`DrawFrame(true)`).
    - If running in Stereo (`both`), it concatenates the left and right images horizontally (`cv::hconcat`).
    - It displays this 2D image in a separate standard OpenCV window using `cv::imshow`.

### 3. Thread Synchronization and Shutdown

```cpp
void Viewer::RequestFinish()
bool Viewer::CheckFinish()
void Viewer::RequestStop()
```
**Explanation:** 
- Because this runs on a separate thread, it cannot be killed abruptly (which would crash the X11/Wayland display server or the OpenGL context).
- These functions use `std::mutex` and `unique_lock` to implement a polite shutdown protocol.
- When `System::Shutdown()` is called, it triggers `RequestFinish()`.
- The `while(1)` loop in `Run()` checks `CheckFinish()` at the bottom of every frame. If true, it breaks the loop, performs cleanup, and signals `SetFinish()`, allowing the main application to finally exit.
