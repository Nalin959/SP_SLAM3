# Documentation: `MapPoint.cc`

## High-Level Overview
The `MapPoint.cc` file implements the `MapPoint` class, which represents a single 3D geometric landmark in the SLAM environment.
As the camera moves, it extracts 2D features (pixels). When the same physical feature is seen from two different camera angles (KeyFrames), the system triangulates those 2D rays to create a 3D `MapPoint`. 
A `MapPoint` is not just a static XYZ coordinate; it is a highly intelligent, self-managing entity. It knows exactly which KeyFrames are currently looking at it, it maintains an aggregate visual descriptor (so it can be matched against new frames), it tracks its own visibility statistics to determine if it is "reliable," and it handles its own destruction or merging when duplicate points are detected.

**Primary Dependencies:**
- `MapPoint.h`, `KeyFrame.h`, `Map.h`, `SPmatcher.h` (for calculating descriptor distances).
- Massive use of `std::mutex` because a MapPoint's properties are constantly queried by the Tracking thread and modified by the Local Mapping thread.

---

## Block-by-Block Breakdown

### 1. Construction and Initialization

```cpp
MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap) : ...
{
    Pos.copyTo(mWorldPos);
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);
    
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}
```
**Explanation:** 
- The standard constructor takes its triangulated 3D position `Pos` and the `KeyFrame` that originally created it (`mpRefKF`).
- It grabs a global map mutex to assign itself a unique ID `mnId`.
- **Note:** In ORB-SLAM3, MapPoints can also be initialized purely from IMU data using an inverse-depth parameterization, which is why the overloaded constructor `MapPoint(const double invDepth, ...)` exists.

### 2. Observation Management

A MapPoint's lifeblood is its "Observations" — the list of KeyFrames that can see it, and the exact 2D pixel index in that KeyFrame.

```cpp
void MapPoint::AddObservation(KeyFrame* pKF, int idx)
{
    // ...
    mObservations[pKF]=indexes;
    if(pKF->mvuRight[idx]>=0) nObs+=2; // Stereo observation counts as 2
    else nObs++;
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
    // ...
    mObservations.erase(pKF);
    if(nObs<=2) bBad=true;
}
```
**Explanation:** 
- The `mObservations` map links a `KeyFrame*` to a tuple `(leftIndex, rightIndex)` representing the feature index in the camera(s).
- **Culling:** If a MapPoint loses its observations (perhaps the local mapping thread realized the triangulation was bad and deleted the connections), and it drops to 2 or fewer observations, it flags itself as `bBad` (dead) because a 3D point needs at least 3 distinct views to be robustly optimized.

### 3. Point Merging (`Replace`)

```cpp
void MapPoint::Replace(MapPoint* pMP)
```
**Explanation:** 
- When the Local Mapping thread detects that two `MapPoint`s actually represent the exact same physical feature in the real world (usually because of a loop closure or overlapping camera frustums), it "merges" them by calling `Replace`.
- This function takes all the observations (KeyFrames) from the *current* MapPoint, forcibly transfers them to the *new* `pMP` MapPoint, and then marks the current MapPoint as `bBad` (dead).

### 4. Visibility and Quality Tracking

```cpp
void MapPoint::IncreaseVisible(int n)
void MapPoint::IncreaseFound(int n)
float MapPoint::GetFoundRatio()
```
**Explanation:** 
- To cull "garbage" points (e.g., points triangulated on the sky, moving cars, or reflections), the MapPoint tracks its success rate.
- `mnVisible`: How many times a camera *should* have seen this point (based on geometry and frustum culling).
- `mnFound`: How many times the feature matching algorithm *actually* found it in the image.
- If the `GetFoundRatio()` drops too low, the Local Mapping thread will permanently delete the point.

### 5. Descriptor Fusion

```cpp
void MapPoint::ComputeDistinctiveDescriptors()
```
**Explanation:** 
- A physical point looks different from different angles and distances. Therefore, the MapPoint actually has *many* 2D descriptors (one from every KeyFrame that observes it).
- When a new camera comes along and wants to track this 3D point, which 2D descriptor should it use for matching?
- This function calculates the pairwise distance between *all* stored descriptors. It selects the one descriptor that has the **least median distance** to all the others. This mathematically ensures the MapPoint uses the most "average" or "representative" visual appearance for tracking.

### 6. Geometric Invariance (Scale and Normal)

```cpp
void MapPoint::UpdateNormalAndDepth()
```
**Explanation:** 
- As a camera moves, a feature changes scale (gets bigger as you move closer) and perspective (skews as you move sideways).
- **Normal Vector:** The MapPoint calculates the average viewing vector from all observing KeyFrames. If a new camera tries to look at this point from an angle greater than ~60 degrees from this normal, the system won't even attempt to match it.
- **Scale Invariance (`mfMinDistance`, `mfMaxDistance`):** Based on the scale pyramid level of the original image feature, the MapPoint calculates the physical depth bounds. If a camera gets closer than `mfMinDistance` or further than `mfMaxDistance`, the feature will be too blurry or too small to reliably track, so the system ignores it.
