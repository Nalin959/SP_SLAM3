# Documentation: `Tracking.cc`

## High-Level Overview
The `Tracking.cc` file implements the **Tracking Thread**, the heart of the SLAM system. It runs synchronously on the main thread and is responsible for processing every single incoming camera frame in real-time.
Its primary goals are to:
1. Extract features from the new image.
2. Estimate the current camera pose (Translation and Rotation) by matching these features against the previously tracked frame or the local 3D map.
3. Decide if the current frame is "important" enough to be promoted to a `KeyFrame` and sent to the Local Mapping thread.

In `SP_SLAM3`, this file has been heavily modified to instantiate the Deep Learning feature extractors (`SPextractor`) instead of the classic ORB extractors, and to utilize `LightGlue` during the critical initialization phase.

**Primary Dependencies:**
- `Tracking.h`
- `SPextractor.h`, `SPmatcher.h` (Neural network feature extraction and matching).
- `LightGlue.h`, `PlaceRecognition.h` (Deep Learning modules).
- `Optimizer.h` (For pose optimization using g2o).
- `PnPsolver.h`, `MLPnPsolver.h` (For camera pose estimation).

---

## Block-by-Block Breakdown

### 1. Initialization and Parameter Loading

```cpp
Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, ...)
bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)
```
**Explanation:** 
- The constructor reads the `.yaml` configuration file to determine the camera model. ORB-SLAM3 natively supports standard `Pinhole` (for narrow-field cameras) and `KannalaBrandt8` (for ultra-wide fisheye cameras).
- It extracts intrinsic calibration parameters ($f_x, f_y, c_x, c_y$) and distortion coefficients ($k_1, k_2, p_1, p_2$).
- **Deep Learning Instantiation:** Inside `ParseORBParamFile`, instead of creating the standard `ORBextractor`, it instantiates the modified `SPextractor`. 
- **Monocular Initialization Edge Case:** In Monocular SLAM, initialization requires finding a Fundamental Matrix between two frames. This requires a very high number of feature matches. Therefore, `mpIniORBextractor` (used only for the first two frames) is instantiated to extract $2 \times N$ features. A strict limit of 2000 features is enforced to ensure it does not exceed `LightGlue`'s maximum tensor size limit (2048 tokens).

### 2. Frame Processing State Machine

*(Note: While the specific lines aren't shown in the snippet, the architectural flow is standard ORB-SLAM3)*

The Tracking thread operates as a finite state machine:
- **`NO_IMAGES_YET`:** Waiting for the first frame.
- **`NOT_INITIALIZED`:** Waiting for enough parallax (movement) between the first two frames to safely triangulate the initial 3D MapPoints. *In `SP_SLAM3`, LightGlue is injected here to vastly improve the success rate of this triangulation.*
- **`OK`:** Standard tracking loop. The system attempts to track the camera using several sequential strategies, falling back to the next if one fails:
  1. `TrackWithMotionModel`: If the camera was moving smoothly, guess its new position based on constant velocity, project the MapPoints, and search locally. (Fastest).
  2. `TrackReferenceKeyFrame`: If motion model fails, match the current frame against the last known KeyFrame using Bag-of-Words. (Slower).
  3. `TrackLocalMap`: Once an initial pose guess is found, project the *entire* local 3D map into the camera and search for more matches to refine the pose using g2o optimization. (Crucial for stability).
- **`LOST`:** If all tracking fails, the system enters Relocalization mode, trying to recognize any part of the global map using the `PlaceRecognition` (NetVLAD) module and Bag-of-Words.

### 3. KeyFrame Decision

```cpp
bool Tracking::NeedNewKeyFrame()
void Tracking::CreateNewKeyFrame()
```
**Explanation:** 
- The system cannot save every single frame; memory would explode, and Bundle Adjustment would grind to a halt.
- `NeedNewKeyFrame` contains heuristics to decide when to drop a breadcrumb. It returns true if:
  - Enough time/frames have passed since the last KeyFrame.
  - The camera has moved a significant physical distance.
  - The number of tracked MapPoints has dropped below a threshold (meaning we are looking at new, unmapped territory).
- If true, `CreateNewKeyFrame` packages the current `Frame` into a `KeyFrame` object and inserts it into the `mpLocalMapper` queue for deep processing.
