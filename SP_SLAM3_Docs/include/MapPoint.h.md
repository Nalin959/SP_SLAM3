# Documentation: `MapPoint.h`

## High-Level Overview
The `MapPoint.h` file defines the `MapPoint` class, which represents a single, physical 3D landmark in the real world (e.g., the corner of a table, a leaf on a tree). 
If `KeyFrame`s are the cameras, `MapPoint`s are what the cameras are looking at. A `MapPoint` is triangulated from 2D pixel observations across multiple frames. 
Because a `MapPoint` is observed from many different angles and distances as the drone flies around, it stores a robust set of metadata: a "Representative Descriptor" (the best neural network embedding out of all the times it was seen), a normal viewing vector, and scale invariance limits (how close/far the drone can get before the feature becomes unrecognizable).

**Primary Dependencies:**
- `KeyFrame.h` (The cameras observing this point).
- `boost/serialization` (To save the 3D point cloud to disk).

---

## Block-by-Block Breakdown

### 1. Serialization (Saving to Disk)

```cpp
template<class Archive> void serialize(Archive & ar, const unsigned int version)
```
**Explanation:** 
- Identical in purpose to the serialization block in `KeyFrame.h`. It writes the 3D position, the deep learning descriptor, and the IDs of all observing KeyFrames to disk so the point cloud can be reconstructed later.

### 2. Physical State Variables

```cpp
cv::Mat mWorldPos;
cv::Mat mNormalVector;
float mfMinDistance;
float mfMaxDistance;
```
**Explanation:** 
- `mWorldPos`: The absolute $(X, Y, Z)$ coordinates of the point in the global map frame.
- `mNormalVector`: The average direction from which this point has been observed. If the drone flies behind the point, it shouldn't try to match it, because it is likely looking at the back of an opaque object.
- `mfMinDistance` / `mfMaxDistance`: The scale invariance limits. The SuperPoint neural network extracts features at multiple scale pyramid levels. These variables define the physical maximum and minimum distances the camera can be from this point and still reliably recognize it.

### 3. Observations and The Covisibility Graph

```cpp
std::map<KeyFrame*, std::tuple<int,int>> mObservations;
void AddObservation(KeyFrame* pKF, int idx);
void EraseObservation(KeyFrame* pKF);
```
**Explanation:** 
- **The Core Link:** `mObservations` maps every `KeyFrame` that can see this point to the exact integer index of the 2D feature in that KeyFrame's `mvKeys` array.
- This is the fundamental data structure that creates the Covisibility Graph. If two KeyFrames are in this map, they are mathematically connected.

### 4. Neural Network Descriptor

```cpp
cv::Mat mDescriptor;
void ComputeDistinctiveDescriptors();
```
**Explanation:** 
- If a point is seen by 10 different KeyFrames, it has 10 different SuperPoint descriptor vectors.
- Storing all 10 is a waste of RAM. `ComputeDistinctiveDescriptors` computes the median descriptor among all observations and stores it as the single, representative `mDescriptor` for this 3D point. This is the descriptor used by the `Tracking` thread when searching for this point in the future.

### 5. Tracking and Culling Counters

```cpp
int mnVisible;
int mnFound;
void SetBadFlag();
void Replace(MapPoint* pMP);
```
**Explanation:** 
- `mnVisible`: The number of times the camera *should* have seen this point (it was inside the camera frustum and within the scale limits).
- `mnFound`: The number of times the neural network *actually* matched the point.
- The `LocalMapping` thread divides these: `mnFound / mnVisible`. If this ratio drops below $25\%$, the system assumes the point was a false positive, a moving object, or a bad triangulation, and triggers `SetBadFlag()`, flagging it for deletion.
- `Replace`: If `LocalMapping` realizes two `MapPoint`s are actually the same physical object, it calls `Replace` to merge them, transferring all `mObservations` from the bad point to the good point.

### 6. Tracking Caches

```cpp
float mTrackProjX, mTrackProjY;
bool mbTrackInView;
```
**Explanation:** 
- These variables act as a temporary cache for the `Tracking` thread. When projecting the 3D map into the live camera feed, the math is calculated once and stored here so it doesn't have to be recalculated multiple times in the same millisecond by different sub-functions.
