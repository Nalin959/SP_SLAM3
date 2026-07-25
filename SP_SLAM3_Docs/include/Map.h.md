# Documentation: `Map.h`

## High-Level Overview
The `Map.h` file defines the `Map` class, which is the central data repository for a single, continuous SLAM session.
A `Map` is primarily composed of two massive sets: all the permanent camera poses (`mspKeyFrames`) and all the physical 3D landmarks (`mspMapPoints`). It acts as the shared memory space for all three main threads (`Tracking`, `LocalMapping`, and `LoopClosing`), requiring strict mutex locking to prevent race conditions (e.g., the Viewer thread trying to draw a MapPoint exactly as the LocalMapping thread deletes it).
In ORB-SLAM3, the architecture was upgraded to support multiple disjoint maps via the `Atlas`. This class represents one of those individual maps.

**Primary Dependencies:**
- `KeyFrame.h`, `MapPoint.h` (The primary entities it stores).
- `boost/serialization` (To save/load the map to disk).
- `<mutex>` (For thread safety).

---

## Block-by-Block Breakdown

### 1. Core Data Structures

```cpp
std::set<MapPoint*> mspMapPoints;
std::set<KeyFrame*> mspKeyFrames;
```
**Explanation:** 
- The heart of the map. These standard `std::set` containers hold pointers to every active `KeyFrame` and `MapPoint` in the current environment. 
- Using `std::set` ensures that no duplicate pointers are accidentally added and allows for relatively fast $O(\log N)$ lookups and deletions.

### 2. Thread Synchronization

```cpp
std::mutex mMutexMap;
std::mutex mMutexMapUpdate;
std::mutex mMutexPointCreation;
```
**Explanation:** 
- Because SLAM is heavily multi-threaded, the Map is constantly under attack.
- `mMutexMap`: Locks the entire map data structure when reading or writing large batches of data.
- `mMutexMapUpdate`: Used specifically by the `Viewer` thread to know when it is safe to pull the latest state for rendering.
- `mMutexPointCreation`: Prevents a subtle bug where the `Tracking` thread and the `LocalMapping` thread might both attempt to triangulate and create a MapPoint at the exact same millisecond, leading to ID conflicts.

### 3. Modifying the Map

```cpp
void AddKeyFrame(KeyFrame* pKF);
void EraseKeyFrame(KeyFrame* pKF);
void ApplyScaledRotation(const cv::Mat &R, const float s, const bool bScaledVel, const cv::Mat t);
```
**Explanation:** 
- Standard CRUD operations to add and remove data.
- `ApplyScaledRotation`: A highly specialized function used during **Map Merging** or **Scale Initialization**. If the system realizes the map is physically tilted relative to real-world gravity, or that its metric scale is off by $20\%$, this function iterates through every single `KeyFrame` and `MapPoint` and applies the $Sim(3)$ transformation to warp the entire map into the correct coordinate frame instantly.

### 4. IMU State Tracking

```cpp
bool mbImuInitialized;
bool mbIsInertial;
bool mbIMU_BA1;
bool mbIMU_BA2;
```
**Explanation:** 
- Visual-Inertial SLAM initializes in stages.
- `mbImuInitialized`: True if the gravity vector and absolute scale have been found.
- `mbIMU_BA1` / `mbIMU_BA2`: Booleans tracking whether the map has undergone the first and second stages of Inertial Bundle Adjustment (which refine the IMU biases). The tracking logic changes its behavior based on how mature the IMU calibration is.

### 5. Serialization and File I/O

```cpp
template<class Archive> void serialize(Archive &ar, const unsigned int version)
void PreSave(std::set<GeometricCamera*> &spCams);
void PostLoad(KeyFrameDatabase* pKFDB, ORBVocabulary* pORBVoc, ...);
```
**Explanation:** 
- `PreSave`: Before writing the `mspKeyFrames` set to disk, the map extracts all the unique `GeometricCamera` models attached to those frames so they can be saved at the top of the file.
- `PostLoad`: After reading the raw data from disk, this function rebuilds all the complex pointer relationships (e.g., reconnecting `KeyFrame`s to their `MapPoint`s using their integer IDs) and re-injects them into the `KeyFrameDatabase` so the loaded map can immediately be used for relocalization.
