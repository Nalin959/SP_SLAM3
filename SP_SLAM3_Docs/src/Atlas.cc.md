# Documentation: `Atlas.cc`

## High-Level Overview
The `Atlas.cc` file implements the `Atlas` class for the SP_SLAM3 system. In the context of ORB-SLAM3 (on which SP_SLAM3 is based), an "Atlas" is a multi-map system. Instead of maintaining a single monolithic map, the system can hold multiple disjoint maps. This is critical for robust SLAM because if tracking is completely lost, the system can spawn a new map in the Atlas and later merge it with older maps if a loop closure or relocalization is detected.

**Primary Dependencies:** 
- `Atlas.h`, `Viewer.h`, `GeometricCamera.h`, `Pinhole.h`, `KannalaBrandt8.h`
- STL containers like `std::set`, `std::vector`, `std::map`.
- Threading dependencies like `std::mutex` and `unique_lock`.

---

## Block-by-Block Breakdown

### Constructors and Destructor

```cpp
Atlas::Atlas(){
    mpCurrentMap = static_cast<Map*>(NULL);
}

Atlas::Atlas(int initKFid): mnLastInitKFidMap(initKFid), mHasViewer(false)
{
    mpCurrentMap = static_cast<Map*>(NULL);
    CreateNewMap();
}
```
**Explanation:** 
- The default constructor initializes the current active map (`mpCurrentMap`) to `NULL`.
- The parameterized constructor accepts an initial KeyFrame ID (`initKFid`), initializes tracking variables, and immediately calls `CreateNewMap()` to instantiate the first map in the Atlas.

```cpp
Atlas::~Atlas()
{
    for(std::set<Map*>::iterator it = mspMaps.begin(), end = mspMaps.end(); it != end;)
    {
        Map* pMi = *it;
        if(pMi)
        {
            delete pMi;
            pMi = static_cast<Map*>(NULL);
            it = mspMaps.erase(it);
        }
        else
            ++it;
    }
}
```
**Explanation:** 
- The destructor is responsible for cleaning up memory. It iterates through the `std::set` of all maps (`mspMaps`) managed by the Atlas. For every valid map pointer, it deletes the map object to prevent memory leaks and erases it from the set.

### Map Management

```cpp
void Atlas::CreateNewMap()
{
    unique_lock<mutex> lock(mMutexAtlas);
    cout << "Creation of new map with id: " << Map::nNextId << endl;
    if(mpCurrentMap){
        cout << "Exits current map " << endl;
        if(!mspMaps.empty() && mnLastInitKFidMap < mpCurrentMap->GetMaxKFid())
            mnLastInitKFidMap = mpCurrentMap->GetMaxKFid()+1; //The init KF is the next of current maximum

        mpCurrentMap->SetStoredMap();
        cout << "Saved map with ID: " << mpCurrentMap->GetId() << endl;
    }
    cout << "Creation of new map with last KF id: " << mnLastInitKFidMap << endl;

    mpCurrentMap = new Map(mnLastInitKFidMap);
    mpCurrentMap->SetCurrentMap();
    mspMaps.insert(mpCurrentMap);
}
```
**Explanation:** 
- This function creates a new independent map within the Atlas.
- It uses a `unique_lock` on `mMutexAtlas` because the map structures are accessed by multiple threads (Tracking, Local Mapping, Loop Closing).
- **State Management:** If an active map already exists (`mpCurrentMap`), it is transition into a "stored" state (`SetStoredMap()`), and the system calculates the starting KeyFrame ID for the new map to ensure global KeyFrame IDs remain sequential. 
- A new `Map` is instantiated, set as the active map, and inserted into the `mspMaps` container.

```cpp
void Atlas::ChangeMap(Map* pMap)
```
**Explanation:** 
- Safely swaps the current active map with a different map pointer. The old map is relegated to storage, and the provided `pMap` is set to active.

### Data Injection & Interaction

```cpp
void Atlas::AddKeyFrame(KeyFrame* pKF)
{
    Map* pMapKF = pKF->GetMap();
    pMapKF->AddKeyFrame(pKF);
}

void Atlas::AddMapPoint(MapPoint* pMP)
{
    Map* pMapMP = pMP->GetMap();
    pMapMP->AddMapPoint(pMP);
}
```
**Explanation:** 
- These are wrapper functions. Rather than adding a KeyFrame or MapPoint directly to the Atlas, the Atlas delegates the object to the specific sub-map that the object belongs to.

### Retrieval and Serialization

```cpp
std::vector<Map*> Atlas::GetAllMaps()
{
    unique_lock<mutex> lock(mMutexAtlas);
    struct compFunctor
    {
        inline bool operator()(Map* elem1 ,Map* elem2)
        {
            return elem1->GetId() < elem2->GetId();
        }
    };
    vector<Map*> vMaps(mspMaps.begin(),mspMaps.end());
    sort(vMaps.begin(), vMaps.end(), compFunctor());
    return vMaps;
}
```
**Explanation:** 
- Returns all maps in the Atlas.
- **Data Structure Optimization:** The maps are stored internally as a `std::set` (which relies on pointer memory addresses for sorting). To return them deterministically, they are dumped into a `std::vector` and sorted sequentially by their internal `GetId()` using a custom inline `compFunctor`.

```cpp
void Atlas::PreSave()
```
**Explanation:** 
- A critical serialization function called before writing the SLAM state to disk.
- It copies all maps into a backup vector (`mvpBackupMaps`) and all camera objects into backup arrays (`mvpBackupCamPin` for Pinhole cameras, `mvpBackupCamKan` for Kannala-Brandt fisheye cameras). 
- It triggers the `PreSave` method on every individual map.

```cpp
void Atlas::PostLoad()
```
**Explanation:** 
- Reconstructs the Atlas state after loading from disk.
- It re-establishes the `mspMaps` set from the loaded backup vectors.
- It triggers the `PostLoad` functionality for every map, feeding them the deserialized Vocabulary and KeyFrameDatabase so they can rebuild their BoW (Bag of Words) vectors for place recognition.

### IMU State Checks
```cpp
bool Atlas::isInertial()
void Atlas::SetInertialSensor()
void Atlas::SetImuInitialized()
bool Atlas::isImuInitialized()
```
**Explanation:** 
- Simple thread-safe getter/setter wrappers that query the `mpCurrentMap` to check if an IMU (Inertial Measurement Unit) is being used and whether the IMU biases have successfully converged (initialized).
