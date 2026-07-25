# Documentation: `Tracking.h`

## High-Level Overview
The `Tracking.h` file defines the most critical and complex thread in the entire SLAM system: The **Tracking Thread**.
While `LocalMapping` and `LoopClosing` run in the background (asynchronously), `Tracking` runs synchronously in the main execution thread. Its primary job is to process every single incoming camera frame in real-time (e.g., 30 FPS). For every frame, it extracts SuperPoint features, associates them with the known 3D map, computes the drone's 6-DOF pose ($X, Y, Z$, Roll, Pitch, Yaw), integrates IMU data, and decides whether the current view is novel enough to be saved as a new `KeyFrame`.

**Primary Dependencies:**
- `SPextractor.h`, `LightGlue.h` (For seeing the world).
- `Optimizer.h` (For computing the math of the camera pose).
- `LocalMapping.h` (To pass new KeyFrames to the background builder).
- `PlaceRecognition.h` (For relocalization if the drone gets lost).

---

## Block-by-Block Breakdown

### 1. The Main Execution API

```cpp
cv::Mat GrabImageStereo(...);
cv::Mat GrabImageRGBD(...);
cv::Mat GrabImageMonocular(...);
void GrabImuData(const IMU::Point &imuMeasurement);
```
**Explanation:** 
- The `System` object routes data into these functions.
- `GrabImuData`: As IMU messages arrive (usually at 200Hz), they are safely pushed into a thread-safe queue (`mlQueueImuData`).
- `GrabImageMonocular`: When a camera frame arrives (at 30Hz), this function wraps the image in a `Frame` object, extracts SuperPoint features, and then calls the private `Track()` function. It blocks until `Track()` is finished and returns the computed camera pose.

### 2. State Machine

```cpp
enum eTrackingState {
    SYSTEM_NOT_READY=-1,
    NO_IMAGES_YET=0,
    NOT_INITIALIZED=1,
    OK=2,
    RECENTLY_LOST=3,
    LOST=4,
    OK_KLT=5
};
eTrackingState mState;
```
**Explanation:** 
- `NOT_INITIALIZED`: The drone is moving, but the system hasn't figured out the initial 3D structure of the world yet (waiting for enough parallax).
- `OK`: Normal tracking. The drone knows exactly where it is.
- `RECENTLY_LOST` / `LOST`: The camera is blocked, the lights went out, or there is too much motion blur. The system stops building the map and frantically tries to figure out where it is by matching the current frame against the `PlaceRecognition` neural network.

### 3. The Core Tracking Functions

```cpp
void Track();
bool TrackWithMotionModel();
bool TrackWithLightGlue();
bool TrackReferenceKeyFrame();
bool Relocalization();
```
**Explanation:** 
- `Track()`: The master switchboard. Based on `mState`, it decides which sub-function to call.
- `TrackWithMotionModel`: The happiest path. Assumes the drone is moving at a constant velocity. It guesses the new pose, projects the local 3D map into the camera, and searches for matches in a tiny pixel radius. Incredibly fast.
- `TrackWithLightGlue`: If the motion model fails (e.g., sudden erratic drone movement), this function uses the heavy LightGlue neural network to brute-force match the current frame against the last known good frame.
- `TrackReferenceKeyFrame`: If all else fails, it tries to match against the nearest `KeyFrame` in the map.
- `Relocalization`: Called if the system is completely `LOST`. Queries the `PlaceRecognition` network to find a visually similar historical KeyFrame, computes a PnP solver, and snaps the drone back onto the known map.

### 4. Local Map Tracking

```cpp
void UpdateLocalMap();
bool TrackLocalMap();
bool NeedNewKeyFrame();
void CreateNewKeyFrame();
```
**Explanation:** 
- After the initial pose is guessed (by `TrackWithMotionModel`, etc.), `UpdateLocalMap()` fetches all `MapPoint`s and `KeyFrame`s physically near the drone.
- `TrackLocalMap()` projects *all* of those local points into the current frame, drastically increasing the number of matches. It then calls `Optimizer::PoseOptimization` to lock in the final, highly accurate camera pose.
- `NeedNewKeyFrame`: Checks if the drone has moved far enough or turned sharply enough since the last KeyFrame. If so, `CreateNewKeyFrame` packages the current `Frame` and hands it off to the `LocalMapping` thread to expand the 3D map.

### 5. IMU Preintegration

```cpp
void PreintegrateIMU();
void PredictStateIMU();
```
**Explanation:** 
- While the camera was waiting for light to hit the sensor, the IMU fired 10 times. `PreintegrateIMU` pulls those 10 IMU messages off the queue and mathematically integrates their accelerations and angular velocities into a single relative motion vector.
- `PredictStateIMU` uses that vector to mathematically guarantee exactly where the camera is *before* a single pixel is processed. This makes Visual-Inertial SLAM almost immune to motion blur and temporary camera blinding.
