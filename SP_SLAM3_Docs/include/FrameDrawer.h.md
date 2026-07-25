# Documentation: `FrameDrawer.h`

## High-Level Overview
The `FrameDrawer.h` file defines a thread-safe helper class responsible for generating the 2D visual overlays used by the `Viewer` thread. 
While the `Tracking` thread is busy extracting features and estimating poses, it occasionally hands a copy of its internal state to the `FrameDrawer`. The `FrameDrawer` then safely isolates this data, allowing the `Viewer` thread to draw the image (complete with colored tracking indicators) at whatever frame rate the user requests, without blocking or slowing down the core SLAM algorithms.

**Primary Dependencies:**
- `Tracking.h` (To copy the current tracking state).
- `MapPoint.h`, `Atlas.h`.
- `<opencv2/core/core.hpp>` (For OpenCV drawing primitives).

---

## Block-by-Block Breakdown

### 1. Data Synchronization

```cpp
void Update(Tracking *pTracker);
std::mutex mMutex;
```
**Explanation:** 
- `Update`: This function is called exclusively by the `Tracking` thread right after it finishes processing a frame. 
- It uses `mMutex` to safely copy all relevant visual data (the raw image, the keypoints, the tracking state) from the live `Tracking` object into the `FrameDrawer`'s internal buffers. 
- By using a mutex here, it guarantees that the `Viewer` thread (which is constantly reading from `FrameDrawer` to render the UI) never reads corrupted, half-written data.

### 2. Rendering Functions

```cpp
cv::Mat DrawFrame(bool bOldFeatures=true);
cv::Mat DrawRightFrame();
```
**Explanation:** 
- These functions are called exclusively by the `Viewer` thread.
- `DrawFrame`: Returns an OpenCV `cv::Mat` (image) ready to be displayed on the screen. It overlays specific visual indicators on top of the raw camera feed:
  - **Green Squares:** Keypoints that are successfully matched to 3D MapPoints and are actively being tracked.
  - **Red Squares:** Keypoints that were tracked recently but lost, or were flagged as outliers by the optimizer.
  - **Blue Lines:** During monocular initialization, it draws lines connecting a feature in Frame 1 to its matched feature in Frame 2.
- `DrawRightFrame`: If running in Stereo (`both == true`), this generates the exact same visual overlays for the right camera feed, which the `Viewer` will horizontally concatenate with the left feed.

### 3. Internal State Buffers

```cpp
cv::Mat mIm, mImRight;
vector<cv::KeyPoint> mvCurrentKeys;
vector<bool> mvbMap, mvbVO;
int mnTracked, mnTrackedVO;
int mState;
```
**Explanation:** 
- These variables hold the cached copy of the `Tracking` thread's state.
- `mvbMap`: A boolean array parallel to `mvCurrentKeys`. If `true`, the corresponding keypoint is matched to a known 3D MapPoint. (Drawn as a Green square).
- `mvbVO`: (Visual Odometry). If `true`, the keypoint is matched to a temporary 3D point that is not yet part of the global map. (Often drawn with a different color/shape).
- `mState`: The current state of the state machine (e.g., `OK`, `LOST`, `NOT_INITIALIZED`). This determines what text to print on the screen.

### 4. Text Overlay Helper

```cpp
void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);
```
**Explanation:** 
- A private helper function that uses OpenCV's `cv::putText` to write the status text in the bottom left corner of the video feed.
- It displays the current `mState` (e.g., "SLAM Mode", "Localization Mode", "LOST"), the number of features currently tracked (`mnTracked`), and whether the IMU is fully initialized.
