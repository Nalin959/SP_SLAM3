# Documentation: `LoopClosing.cc`

## High-Level Overview
The `LoopClosing.cc` file implements the `LoopClosing` class, which is arguably the most critical component for long-term SLAM accuracy. It runs in its own background thread.
As a camera moves through an environment, tiny tracking errors accumulate, leading to "drift." If the camera returns to a previously visited location, the system uses Loop Closing to recognize the place, calculate the accumulated drift, and forcibly warp the entire trajectory and 3D map back into alignment (Loop Correction). 
In ORB-SLAM3, this thread is also responsible for **Map Merging**: if the tracking gets completely lost, the system starts a *new* local map. When it eventually recognizes a place from an *old* map, this thread seamlessly sews the two disconnected maps together.

**Primary Dependencies:**
- `LoopClosing.h`, `Sim3Solver.h`, `SPmatcher.h`
- `KeyFrameDatabase.h` (for Place Recognition querying).
- `Optimizer.h` (for Pose Graph Optimization and Global Bundle Adjustment).

---

## Block-by-Block Breakdown

### 1. The Main Loop (`Run()`)

```cpp
void LoopClosing::Run()
{
    while(1)
    {
        if(CheckNewKeyFrames())
        {
            if(NewDetectCommonRegions())
            {
                if(mbMergeDetected)
                {
                    // ... Perform Map Merging ...
                    MergeLocal(); // or MergeLocal2() for IMU
                }
                
                if(mbLoopDetected)
                {
                    // ... Perform Loop Correction ...
                    CorrectLoop();
                }
            }
        }
        // ...
    }
}
```
**Explanation:** 
- The thread continuously checks a queue of KeyFrames provided by the Local Mapping thread.
- For every KeyFrame, it asks `NewDetectCommonRegions()` if this image looks like somewhere we have been before.
- If it detects a match with a different map, it triggers a **Merge**.
- If it detects a match within the current map, it triggers a **Loop Closure**.

### 2. Detecting Common Regions (Place Recognition)

```cpp
bool LoopClosing::NewDetectCommonRegions()
{
    // ...
    if (mpPlaceRecognition && mpPlaceRecognition->isLoaded() && mpCurrentKF->mGlobalDescriptor.numel() > 0)
    {
        // Learned place recognition path (NetVLAD/CosPlace)
        auto vpCandidates = mpPlaceRecognition->query(...);
    }
    
    // Always query DBoW3 as well
    mpKeyFrameDB->DetectNBestCandidates(mpCurrentKF, vpLoopBowCandExtra, vpMergeBowCandExtra, 3);
```
**Explanation:** 
- **Hybrid Place Recognition:** The system uses two parallel systems to find candidate matches.
  1. Traditional Bag-of-Words (`DBoW3`) based on local features.
  2. Optional Deep Learning global descriptors (like NetVLAD or CosPlace) passed via the `mpPlaceRecognition` pointer, which are vastly superior at handling severe illumination/seasonal changes.
- It splits the candidates into `vpLoopBowCand` (same map) and `vpMergeBowCand` (different map).

### 3. Geometric Validation (The Sim3 Pipeline)

```cpp
bool LoopClosing::DetectCommonRegionsFromBoW(...)
{
    // ... loop through candidates ...
    int numBoWMatches = matcherBoW.SearchByBoW(mpCurrentKF, vpCovKFi[j], vvpMatchedMPs[j]);
    
    if(numBoWMatches >= nBoWMatches)
    {
        Sim3Solver solver = Sim3Solver(mpCurrentKF, pMostBoWMatchesKF, vpMatchedPoints, bFixedScale, vpKeyFrameMatchedMP);
        // ... RANSAC loop ...
        mTcm = solver.iterate(20, bNoMore, vbInliers, nInliers, bConverge);
```
**Explanation:** 
- A high appearance similarity score is not enough (it could be two identical-looking doors in different hallways). 
- To prove a true loop closure, the geometry must perfectly align.
- **Why Sim3?** Over long distances, monocular SLAM suffers from *scale drift* (the map shrinks or grows). A standard SE(3) transformation (Rotation + Translation) cannot fix this. A **Sim3** transformation (Similarity SE(3) = Rotation + Translation + Scale) calculates exactly how much the scale drifted.
- The `Sim3Solver` uses RANSAC with Horn's Method to find the Sim3 transform `mTcm` that perfectly aligns the 3D MapPoints of the current frame with the candidate frame.

### 4. Advanced Inlier Refinement

```cpp
        if(bConverge)
        {
            // Project all candidate points into current frame using the new Sim3
            int numProjMatches = matcher.SearchByProjection(mpCurrentKF, mScw,  mvpLoopMapPoints, mvpCurrentMatchedPoints, 10);
            
            if(numProjMatches >= nProjMatches)
            {
                // Non-linear Optimization
                int numOptMatches = Optimizer::OptimizeSim3(mpCurrentKF, pKFi, vpMatchedMP, gScm, 10, mbFixScale, ...);
```
**Explanation:** 
- If RANSAC finds a Sim3 transform, the system rigorously tests it.
- It projects the *entire* local 3D point cloud of the candidate into the current camera view using the Sim3 transform, and looks for pixel-perfect matches (`SearchByProjection`).
- If enough matches are found, it runs a non-linear optimizer (`Optimizer::OptimizeSim3`) to finely tune the Sim3 matrix. If the inlier count remains high, the loop closure is mathematically verified.

### 5. Loop Correction and Map Merging (Triggering)

*(Note: The actual heavy lifting of `CorrectLoop()` and `MergeLocal()` is massive and handled in their respective functions further down the file, but the trigger logic is here).*

```cpp
// Inside Run() - For loops
mg2oLoopScw = mg2oLoopSlw;
if(mpCurrentKF->GetMap()->IsInertial())
{
    // If inertial, force only yaw
    phi(0)=0; phi(1)=0;
    g2oSww_new = g2o::Sim3(ExpSO3(phi),g2oSww_new.translation(),1.0);
}
CorrectLoop();
```
**Explanation:** 
- **IMU Yaw Correction constraint:** In Visual-Inertial SLAM, gravity provides an absolute reference for Pitch and Roll. Therefore, drift in Pitch and Roll is physically impossible over the long term (unlike scale or Yaw).
- When applying the Sim3 correction in VI-SLAM, the system explicitly forces the Pitch and Roll correction to zero (`phi(0)=0`, `phi(1)=0`), ensuring the loop closure doesn't artificially tilt the gravity vector.
- Finally, it calls `CorrectLoop()` or `MergeLocal()`, which will lock down the entire system, fuse duplicate points, propagate the Sim3 correction down the Spanning Tree, and launch an asynchronous **Global Bundle Adjustment (GBA)** thread to flawlessly smooth out the rest of the map.
