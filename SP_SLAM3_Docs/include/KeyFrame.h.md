# Documentation: `KeyFrame.h`

## High-Level Overview
The `KeyFrame.h` file defines the `KeyFrame` class, which is arguably the most structurally important data structure in the entire SLAM system.
While a standard `Frame` is ephemeral (it exists only for a fraction of a second to track movement and is then immediately deleted to save RAM), a **KeyFrame** is permanent. It is a curated snapshot of the environment that is permanently inserted into the `Map`.
KeyFrames are the nodes in the global SLAM graph. The system relies on KeyFrames to perform Bundle Adjustment (to refine the map), Loop Closing (to correct long-term drift), and Map Merging (when multiple SLAM sessions connect). 

Because they are permanent, `KeyFrame`s contain extensive metadata about their relationships with other KeyFrames (Covisibility Graphs, Spanning Trees) and massive amounts of serialization boilerplate for saving maps to disk.

**Primary Dependencies:**
- `Frame.h` (A KeyFrame is initialized from a standard Frame).
- `MapPoint.h` (The 3D points observed by this KeyFrame).
- `boost/serialization` (Extensive logic for Map Saving/Loading).
- `DBoW3` (Bag of Words for loop closure).

---

## Block-by-Block Breakdown

### 1. Serialization (Saving to Disk)

```cpp
template<class Archive> void serializeMatrix(...)
template<class Archive> void serializeVectorKeyPoints(...)
template<class Archive> void serialize(Archive& ar, const unsigned int version)
```
**Explanation:** 
- The first 200+ lines of this file are entirely dedicated to `boost::serialization`. 
- To save a SLAM session, the system must write every `KeyFrame` to the hard drive. 
- The `serialize` function meticulously writes out over 50 individual variables, including the camera pose, the deep learning descriptors, the IMU biases, and the 2D keypoints. It also writes out the unique ID integers of the connected MapPoints so the graph can be stitched back together when loaded.

### 2. Constructors & Core State

```cpp
KeyFrame(Frame &F, Map* pMap, KeyFrameDatabase* pKFDB);
cv::Mat GetPose();
cv::Mat GetCameraCenter();
```
**Explanation:** 
- A `KeyFrame` is born by copying a live `Frame`. It strips away the raw image pixels but keeps the neural network features, descriptors, and current estimated pose.
- Unlike a `Frame`, a `KeyFrame`'s pose can be modified at any time by the `LocalMapping` or `LoopClosing` threads running in the background as they optimize the map. Therefore, accessing the pose (`GetPose()`) is protected by a mutex (`mMutexPose`).

### 3. The Covisibility Graph

```cpp
void AddConnection(KeyFrame* pKF, const int &weight);
void UpdateConnections(bool upParent=true);
std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
std::map<KeyFrame*,int> mConnectedKeyFrameWeights;
```
**Explanation:** 
- The **Covisibility Graph** is the backbone of the SLAM map. If KeyFrame A and KeyFrame B both observe the same physical 3D MapPoints, they are "connected" in this graph.
- The `weight` of the connection is the exact number of 3D MapPoints they share.
- The `LocalMapping` thread heavily uses this. When optimizing the position of KeyFrame A, it only pulls in the highly-connected neighbors (the "Local Window") to keep the Bundle Adjustment math fast and bounded.

### 4. The Spanning Tree

```cpp
void AddChild(KeyFrame* pKF);
void ChangeParent(KeyFrame* pKF);
KeyFrame* mpParent;
std::set<KeyFrame*> mspChildrens;
```
**Explanation:** 
- The **Essential Graph** (or Spanning Tree) is a subset of the Covisibility Graph used for lightning-fast Pose Graph Optimization (PGO) during Loop Closures.
- Every KeyFrame has exactly one Parent (the KeyFrame that shared the most points with it when it was created), ensuring a minimally connected, loop-free tree structure across the entire map.

### 5. Loop and Merge Variables

```cpp
cv::Mat mTcwGBA;
IMU::Bias mBiasGBA;
cv::Mat mTcwMerge;
```
**Explanation:** 
- When a massive loop closure occurs, a Global Bundle Adjustment (GBA) is launched in a background thread that might take 10 seconds to compute. 
- During those 10 seconds, the live tracking continues. `mTcwGBA` and `mTcwMerge` act as temporary holding buffers. Once the background math finishes, the system carefully merges the GBA corrections into the live tracking state using these buffer variables without crashing the drone.

### 6. MapPoint Associations

```cpp
std::vector<MapPoint*> mvpMapPoints;
void AddMapPoint(MapPoint* pMP, const size_t &idx);
void ReplaceMapPointMatch(const int &idx, MapPoint* pMP);
```
**Explanation:** 
- An array storing pointers to the 3D MapPoints corresponding to each 2D feature.
- `ReplaceMapPointMatch`: When the system detects that two separate 3D points are actually the same physical object (e.g., viewing a corner from two different angles), it merges them and uses this function to update all KeyFrames to point to the surviving MapPoint.
