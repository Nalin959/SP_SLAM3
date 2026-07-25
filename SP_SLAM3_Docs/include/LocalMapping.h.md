# Documentation: `LocalMapping.h`

## High-Level Overview
The `LocalMapping.h` file declares the `LocalMapping` class, which operates as its own independent thread in the SLAM architecture.
While the `Tracking` thread handles high-speed, real-time pose estimation frame-by-frame, the `LocalMapping` thread runs asynchronously in the background. Its job is to maintain the quality of the map. When the `Tracking` thread decides a frame is important enough to become a `KeyFrame`, it throws it into a queue for `LocalMapping`.
`LocalMapping` pulls these KeyFrames from the queue, triangulates new 3D MapPoints, deletes bad MapPoints, culls redundant KeyFrames, and most importantly, runs **Local Bundle Adjustment (BA)** to mathematically refine the map and the camera trajectory.

It also contains the highly complex logic required to initialize the IMU (Gravity, Scale, and Biases) in Visual-Inertial mode.

**Primary Dependencies:**
- `Tracking.h`, `LoopClosing.h` (The other main threads it must synchronize with).
- `KeyFrame.h`, `MapPoint.h` (The entities it optimizes).

---

## Block-by-Block Breakdown

### 1. Thread Execution and Queuing

```cpp
void Run();
void InsertKeyFrame(KeyFrame* pKF);
std::list<KeyFrame*> mlNewKeyFrames;
```
**Explanation:** 
- `Run()`: The infinite loop executing in the background thread. It constantly checks `mlNewKeyFrames`.
- `InsertKeyFrame()`: Called by the `Tracking` thread to push a new KeyFrame into the queue.
- If the queue is empty, `LocalMapping` sleeps to save CPU cycles.

### 2. Map Maintenance Pipeline

```cpp
void ProcessNewKeyFrame();
void CreateNewMapPoints();
void MapPointCulling();
void SearchInNeighbors();
void KeyFrameCulling();
```
**Explanation:** 
- This is the sequence of operations `LocalMapping` performs on every new KeyFrame:
  1. **ProcessNewKeyFrame**: Computes the Bag of Words vector and updates the Covisibility Graph connections.
  2. **MapPointCulling**: Iterates through recent MapPoints. If a point was triangulated but rarely seen again (e.g., it was noise or a moving car), it is aggressively deleted.
  3. **CreateNewMapPoints**: Triangulates new 3D points by finding matches between the new KeyFrame and its most connected neighbors (using Epipolar Geometry).
  4. **SearchInNeighbors**: Fuses redundant points (if two cameras triangulated the exact same physical corner slightly differently, they are merged).
  5. **Local Bundle Adjustment**: (Called internally) Optimizes the local window of KeyFrames using g2o.
  6. **KeyFrameCulling**: If $90\%$ of the points in a KeyFrame are also seen by other KeyFrames (e.g., the drone was hovering), the KeyFrame is deemed redundant and deleted to keep the map lightweight.

### 3. Thread Synchronization

```cpp
void RequestStop();
bool Stop();
void InterruptBA();
```
**Explanation:** 
- `LocalMapping` and `LoopClosing` cannot modify the map at the same time. 
- If `LoopClosing` detects a massive loop and wants to warp the entire map, it calls `RequestStop()` on `LocalMapping`.
- Bundle Adjustment can take several seconds. If a stop is requested, `InterruptBA()` allows `LocalMapping` to abort its math mid-calculation and yield the lock to `LoopClosing`.

### 4. Visual-Inertial Initialization

```cpp
void InitializeIMU(float priorG = 1e2, float priorA = 1e6, bool bFirst = false);
void ScaleRefinement();
bool bInitializing;
```
**Explanation:** 
- When running Monocular-Inertial SLAM, the system initially does not know its physical scale (a tiny toy house close up looks identical to a real house far away).
- `InitializeIMU`: After a few seconds of movement, this function sets up a massive optimization problem to jointly solve for:
  1. The true direction of Gravity.
  2. The absolute metric Scale of the map.
  3. The initial Gyroscope and Accelerometer biases.
- If the drone doesn't accelerate enough, this initialization fails. Once successful, the map becomes strictly metric (1 unit = 1 meter).
