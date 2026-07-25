# Documentation: `MapDrawer.cc`

## High-Level Overview
The `MapDrawer.cc` file implements the `MapDrawer` class, which acts as the bridge between the internal SLAM data structures (`Map`, `Atlas`, `KeyFrame`, `MapPoint`) and the **Pangolin 3D Visualization engine**. 
It is executed in a completely separate UI thread. Its sole purpose is to read the state of the SLAM system in real-time and issue raw OpenGL commands (`glVertex3f`, `glBegin`, etc.) to draw the point cloud, the camera trajectory, the active map, and the background stored maps.

**Primary Dependencies:**
- `MapDrawer.h`, `MapPoint.h`, `KeyFrame.h`
- Pangolin (`<pangolin/pangolin.h>`) and OpenGL.
- Threading (`std::mutex`) because the UI thread is reading memory that the Local Mapping and Tracking threads are simultaneously writing to.

---

## Block-by-Block Breakdown

### 1. Initialization and Configuration

```cpp
MapDrawer::MapDrawer(Atlas* pAtlas, const string &strSettingPath):mpAtlas(pAtlas)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    bool is_correct = ParseViewerParamFile(fSettings);
    // ...
}
```
**Explanation:** 
- The class takes a pointer to the global `Atlas` (the container of all maps).
- It parses the YAML settings file (e.g., `EuRoC.yaml`) to load cosmetic visualization parameters: `Viewer.KeyFrameSize`, `Viewer.PointSize`, line widths, etc.

### 2. Drawing the 3D Point Cloud

```cpp
void MapDrawer::DrawMapPoints()
{
    const vector<MapPoint*> &vpMPs = mpAtlas->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = mpAtlas->GetReferenceMapPoints();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0); // Black

    for(size_t i=0; i<vpMPs.size(); i++)
    {
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i])) continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
    }
    glEnd();
    
    // ... Draws Reference MapPoints in Red ...
}
```
**Explanation:** 
- Iterates through the entire point cloud of the active map.
- **Color Coding:** 
  - The vast majority of points (historical points) are drawn in **Black**.
  - Points in the "Reference" set (points actively seen by the current local window of cameras) are drawn in **Red**. This gives the user a visual cue of exactly what part of the map the camera is currently interacting with.
- It skips any point flagged as `isBad()` to avoid rendering phantom geometry that the Local Mapping thread recently deleted.

### 3. Drawing the Trajectories and Graphs

```cpp
void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph)
{
    const vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    // ...
    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // ... OpenGL Matrix transformations ...
            glMultMatrixf(Twc.ptr<GLfloat>(0));
            // ... Draw a 3D frustum (pyramid) representing the camera ...
```
**Explanation:** 
- For every KeyFrame, it extracts the camera-to-world pose `Twc`. It loads this into OpenGL's matrix stack, allowing it to draw a simple 3D pyramid (frustum) representing the physical camera in the world.
- **Color Coding by Map:** Notice the use of `pKF->mnOriginMapId`. If the Atlas contains multiple merged maps, each map is assigned a different color (`mfFrameColors`). This visually demonstrates to the user how different disconnected maps were sewn together.

```cpp
    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f); // Translucent Green
        glBegin(GL_LINES);
        
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            // ... draw line between Ow and Ow2 ...
            
            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();
            // ... draw line ...
            
            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            // ... draw line ...
```
**Explanation:** 
- Visualizes the mathematical structure of the SLAM system.
- It draws the dense **Covisibility Graph** (cameras that look at the same things), the minimal **Spanning Tree** (the rigid hierarchical backbone), and explicit **Loop Closure Edges**.
- **Inertial Graph:** If `bDrawInertialGraph` is true, it draws red lines connecting sequential KeyFrames, representing the temporal IMU preintegration links.

### 4. Updating the Live Camera

```cpp
void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
```
**Explanation:** 
- The Tracking thread rapidly calls `SetCurrentCameraPose` at 30+ FPS. 
- The UI thread uses `DrawCurrentCamera` to render a thick green frustum representing the live, immediate state of the camera, floating dynamically over the static, permanent KeyFrame trajectory.
- `GetCurrentOpenGLCameraMatrix` handles the math of converting the OpenCV `cv::Mat` pose (which uses computer vision conventions: Z-forward, Y-down) into a Pangolin `OpenGlMatrix` (which uses OpenGL conventions).
