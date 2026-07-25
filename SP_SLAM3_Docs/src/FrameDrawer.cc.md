# Documentation: `FrameDrawer.cc`

## High-Level Overview
The `FrameDrawer.cc` file implements the `FrameDrawer` class, which acts as the visual rendering engine for the SLAM system. While the core SLAM threads (Tracking, Local Mapping, Loop Closing) run invisibly in the background, the `FrameDrawer` provides real-time visual feedback to the user or developer. 
It takes the raw video frames and overlays them with graphical debugging information, such as extracted features, triangulated map points, and the current state of the SLAM state machine (e.g., "INITIALIZING", "TRACKING", "LOST").

**Primary Dependencies:**
- `FrameDrawer.h`, `Tracking.h`, `OpenCVCompat.h`
- OpenCV GUI and drawing modules (`cv::circle`, `cv::rectangle`, `cv::line`, `cv::putText`)
- Standard Threading (`std::mutex`, `std::unique_lock`)

---

## Block-by-Block Breakdown

### 1. State Synchronization & Mutex Locking

```cpp
void FrameDrawer::Update(Tracking *pTracker)
{
    unique_lock<mutex> lock(mMutex);
    pTracker->mImGray.copyTo(mIm);
    mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;
    // ...
    mState=static_cast<int>(pTracker->mLastProcessedState);
}
```
**Explanation:** 
- The SLAM system runs asynchronously. The Tracking thread updates at ~30 FPS, while the Viewer thread (which calls `FrameDrawer`) updates independently at whatever frame rate the monitor/GUI can handle.
- To prevent data races (where the Viewer tries to read a frame while the Tracker is overwriting it), `Update()` uses a `std::unique_lock` on `mMutex`.
- It deeply copies (`copyTo()`) the latest image, the 2D keypoints, the tracking state (`mState`), and flags indicating whether each point is an established map point (`mvbMap`) or a newly created visual odometry point (`mvbVO`).

### 2. Main Drawing Function

```cpp
cv::Mat FrameDrawer::DrawFrame(bool bOldFeatures)
{
    // ... setup and locking ...
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        // copy cached variables to local scope to minimize mutex hold time
    }
```
**Explanation:** 
- `DrawFrame()` generates the final annotated image for the left (or monocular) camera. 
- **Performance Optimization:** Notice that it locks the mutex, copies the necessary data into local function variables, and immediately releases the lock before doing any heavy OpenCV drawing. This ensures the Tracking thread is never blocked by the Viewer rendering geometry.

### 3. Rendering Initialization State

```cpp
if(state==Tracking::NOT_INITIALIZED)
{
    for(unsigned int i=0; i<vMatches.size(); i++)
    {
        if(vMatches[i]>=0)
        {
            cv::line(im,vIniKeys[i].pt,vCurrentKeys[vMatches[i]].pt, cv::Scalar(0,255,0));
        }
    }
}
```
**Explanation:** 
- If the SLAM system is still attempting to initialize (e.g., it hasn't seen enough parallax to create the first 3D map), it draws green lines connecting the features from the very first frame (`vIniKeys`) to the current frame (`vCurrentKeys`). 
- This visualizes the optical flow and helps the user understand if they are moving the camera correctly to induce parallax.

### 4. Rendering Active Tracking State

```cpp
else if(state==Tracking::OK && bOldFeatures) //TRACKING
{
    // ...
    for(int i=0;i<n;i++)
    {
        if(vbVO[i] || vbMap[i])
        {
            // Calculate bounding box for a square
            pt1.x=vCurrentKeys[i].pt.x-r; pt1.y=vCurrentKeys[i].pt.y-r;
            pt2.x=vCurrentKeys[i].pt.x+r; pt2.y=vCurrentKeys[i].pt.y+r;

            if(vbMap[i]) {
                cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0)); // GREEN
                cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(0,255,0),-1);
            }
            else {
                cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0)); // BLUE
                cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(255,0,0),-1);
            }
        }
    }
}
```
**Explanation:** 
- Once the map is initialized and the system is actively tracking, it draws geometric markers over the successfully matched features.
- **Green Squares/Dots (`vbMap`):** These are mature `MapPoints` that have been triangulated and exist in the global map. They are the backbone of the tracking stability.
- **Blue Squares/Dots (`vbVO`):** These are "Visual Odometry" points. They are temporary points triangulated from recent frames that haven't yet been promoted to the global map. They help maintain tracking when turning corners into unknown territory.

### 5. Reprojection Error Visualization

```cpp
else if(state==Tracking::OK && !bOldFeatures)
{
    // ...
    if(mProjectPoints.find(mp_id) != mProjectPoints.end())
    {
        cv::Point2f p_proj = mMatchedInImage[mp_id];
        cv::line(im, p_proj, p_image, cv::Scalar(0, 255, 0), 2);
    }
    // ...
    cv::line(im,cv::Point2f(u, v), point_im,cv::Scalar(0, 0, 255), 1); // Red line for outliers
}
```
**Explanation:** 
- This is an alternative visualization mode (toggled by `bOldFeatures`). 
- Instead of just drawing dots, it draws a line connecting the *predicted* location of a 3D point (where the camera model says it should be) and the *actual* matched 2D location (where the neural network/feature extractor found it).
- **Green Lines:** Successful inliers. The length of the line represents the reprojection error.
- **Red Lines:** Outliers. The point was found, but the distance between the prediction and the reality was too large, so the SLAM system rejected it.

### 6. Text Overlay (HUD)

```cpp
void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
```
**Explanation:** 
- Expands the bottom of the image matrix by allocating a new, slightly taller matrix (`cv::Mat(im.rows+textSize.height+10, im.cols, ...)`).
- Uses `cv::putText` to render a Heads-Up Display (HUD) showing the current state (`TRYING TO INITIALIZE`, `TRACKING`, `LOST`) along with real-time statistics queried from the `Atlas` (Number of Maps, KeyFrames, MapPoints, and current active matches).
