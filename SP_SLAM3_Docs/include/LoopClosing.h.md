# Documentation: `LoopClosing.h`

## High-Level Overview
The `LoopClosing.h` file declares the third major thread in the SLAM architecture: the `LoopClosing` thread.
In any SLAM system, small numerical errors in tracking accumulate over time. If a drone flies in a large circle for 10 minutes, it might return to its starting position, but the SLAM math will tell it that it is 5 meters away. This is called "Drift".
The `LoopClosing` thread runs continuously in the background, analyzing every new `KeyFrame` using the `KeyFrameDatabase` and `PlaceRecognition`. When it mathematically proves the drone is looking at a scene it has visited before, it triggers a **Loop Closure**. It computes the exact amount of accumulated drift (represented as a $Sim(3)$ transformation to account for scale drift in monocular setups) and violently warps the entire map backward in time to snap everything perfectly into place, permanently eliminating the drift.

It also handles **Map Merging**—if tracking was lost and a new sub-map was started, this thread stitches the new map back onto the old map once it recognizes a shared landmark.

**Primary Dependencies:**
- `PlaceRecognition.h`, `KeyFrameDatabase.h` (To find loops).
- `LocalMapping.h`, `Tracking.h` (To pause them while warping the map).
- `g2o::Sim3` (3D Similarity transformations: Rotation, Translation, and Scale).

---

## Block-by-Block Breakdown

### 1. Main Execution Loop

```cpp
void Run();
void InsertKeyFrame(KeyFrame *pKF);
std::list<KeyFrame*> mlpLoopKeyFrameQueue;
```
**Explanation:** 
- `Run()`: The infinite loop executing in the background thread.
- `InsertKeyFrame()`: Called by the `LocalMapping` thread once it finishes triangulating and polishing a new KeyFrame. The KeyFrame is placed in the `mlpLoopKeyFrameQueue` for inspection.

### 2. Place Recognition and Similarity Math

```cpp
bool NewDetectCommonRegions();
bool DetectAndReffineSim3FromLastKF(...);
int FindMatchesByProjection(...);
```
**Explanation:** 
- **`NewDetectCommonRegions()`**: Pops a KeyFrame off the queue and queries the `PlaceRecognition` module. If `PlaceRecognition` (e.g., using NetVLAD or DBoW) flags a historical KeyFrame as a match, the system enters verification mode.
- **`DetectAndReffineSim3FromLastKF()`**: Two images looking similar isn't enough; they must be geometrically consistent. This function calculates the $Sim(3)$ transformation (Rotation, Translation, Scale) between the current KeyFrame and the historical KeyFrame. If this $Sim(3)$ projection yields a high number of physically accurate 3D inlier points (`FindMatchesByProjection`), the loop closure is verified and locked in.

### 3. Loop Correction and Merging

```cpp
void CorrectLoop();
void MergeLocal();
```
**Explanation:** 
- **`CorrectLoop()`**: The most critical function. It asks `LocalMapping` to pause. It computes the mathematical error (Drift). It then propagates that error backward through the Essential Graph (Spanning Tree), smoothly warping the entire map and camera trajectory to close the physical gap. Finally, it merges the duplicated 3D MapPoints at the closure seam.
- **`MergeLocal()`**: Performs the same mathematical warping, but instead of bending a single map into a loop, it takes an active map and a dormant map and snaps them together into a unified coordinate system.

### 4. Global Bundle Adjustment (GBA)

```cpp
void RunGlobalBundleAdjustment(Map* pActiveMap, unsigned long nLoopKF);
bool isRunningGBA();
std::thread* mpThreadGBA;
```
**Explanation:** 
- `CorrectLoop()` performs a fast, rigid graph optimization to snap the map together quickly so the drone doesn't crash while waiting.
- However, to make the map perfectly mathematically optimal, a **Global Bundle Adjustment (GBA)** must be run. GBA optimizes *every single pose and point* in the entire map simultaneously. 
- Because GBA can take 10+ seconds on a large map, `RunGlobalBundleAdjustment` is spawned on a 4th, entirely separate background thread (`mpThreadGBA`). While GBA runs, live tracking and mapping continue unharmed.

### 5. Loop State Variables

```cpp
bool mbLoopDetected;
g2o::Sim3 mg2oLoopScw;
std::vector<MapPoint*> mvpLoopMPs;
```
**Explanation:** 
- State variables populated when a loop is found. `mg2oLoopScw` is the computed $Sim(3)$ correction pose, and `mvpLoopMPs` are the 3D points that bridge the gap between the start and end of the loop.
