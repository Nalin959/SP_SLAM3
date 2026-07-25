# Documentation: `Atlas.h`

## High-Level Overview
The `Atlas.h` header file defines the `Atlas` class, which is the foundational data structure that enables the multi-map capability of ORB-SLAM3 (and by extension `SP_SLAM3`). 
In earlier SLAM versions (like ORB-SLAM2), if the camera was covered or tracking was completely lost, the entire SLAM system would fail and reset. The `Atlas` solves this by maintaining a collection (a "forest") of disconnected `Map`s. If tracking is lost, it simply puts the current Map on hold and starts building a completely new `Map`. If the camera later sees a location from an old Map, the Loop Closing thread merges the two Maps together.

**Primary Dependencies:**
- `Map.h`, `MapPoint.h`, `KeyFrame.h` (The building blocks of a single map).
- `GeometricCamera.h`, `Pinhole.h`, `KannalaBrandt8.h` (Camera models stored globally).
- `boost/serialization/...` (For saving the entire Atlas state to disk).

---

## Block-by-Block Breakdown

### 1. Boost Serialization Interface

```cpp
class Atlas {
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
```
**Explanation:** 
- The `Atlas` is the root node when saving a SLAM session to disk. By serializing the `Atlas`, we serialize every Map, KeyFrame, and MapPoint inside it.
- **Boost Bug Workaround:** Note the comments regarding `std::set`. Due to a known bug in Boost 1.58 on Ubuntu 16.04, the developers explicitly copy `mspMaps` into a `std::vector` (`mvpBackupMaps`) before serializing, rather than serializing the `std::set` directly.
- **Static IDs:** The static ID counters (`Map::nNextId`, `Frame::nNextId`) are serialized here. This ensures that when a map is loaded from disk, the system doesn't accidentally assign a new KeyFrame an ID that already exists.

### 2. Multi-Map Management

```cpp
void CreateNewMap();
void ChangeMap(Map* pMap);
int CountMaps();
Map* GetCurrentMap();
```
**Explanation:** 
- These functions manage the state machine of the multi-map system.
- `CreateNewMap()` is called by the Tracking thread when it gets hopelessly lost and needs to start fresh.
- `mpCurrentMap` points to the map the Tracking thread is currently building. All operations (adding KeyFrames, optimizing) default to this map.

### 3. API Pass-through to Current Map

```cpp
void AddKeyFrame(KeyFrame* pKF);
void AddMapPoint(MapPoint* pMP);
void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
std::vector<KeyFrame*> GetAllKeyFrames();
```
**Explanation:** 
- To avoid breaking compatibility with ORB-SLAM2 code that expected a single `Map` object, the `Atlas` provides pass-through functions. Calling `Atlas::AddKeyFrame()` simply calls `mpCurrentMap->AddKeyFrame()`.

### 4. Global State Tracking

```cpp
std::set<Map*> mspMaps;
std::set<Map*> mspBadMaps;
std::vector<GeometricCamera*> mvpCameras;
```
**Explanation:** 
- `mspMaps` holds every active map in the system.
- `mspBadMaps` holds maps that have been discarded (e.g., if a new map was started, but tracking was lost after only 2 frames, that map is considered "bad" and thrown away).
- `mvpCameras` stores the physical camera definitions (intrinsics, distortion). Storing them at the Atlas level ensures that even if maps are merged, they reference the correct physical sensor models.

### 5. IMU State

```cpp
bool isInertial();
void SetInertialSensor();
void SetImuInitialized();
bool isImuInitialized();
```
**Explanation:** 
- The Atlas tracks whether the system is running Visual-Inertial SLAM. 
- `SetImuInitialized()` is a critical state flag. IMU initialization requires a few seconds of movement to correctly estimate scale, gravity direction, and sensor biases. Until the Atlas marks the IMU as initialized, the system relies purely on vision.
