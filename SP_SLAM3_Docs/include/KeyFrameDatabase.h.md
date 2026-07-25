# Documentation: `KeyFrameDatabase.h`

## High-Level Overview
The `KeyFrameDatabase.h` file defines a highly optimized data structure designed for **Place Recognition**.
In SLAM, it is critical to recognize when the drone has returned to a location it has seen before. This allows the system to close a loop (canceling out accumulated drift) or relocalize (if tracking was temporarily lost). 
Comparing the current camera frame against every single historical `KeyFrame` sequentially would be computationally impossible in real-time. To solve this, the `KeyFrameDatabase` uses a Bag-of-Words (BoW) **Inverted File Index**. This acts like a search engine for images, allowing the system to instantly query the database for the top $N$ most visually similar past KeyFrames in constant $O(1)$ time.

**Primary Dependencies:**
- `SPVocabulary.h` (The pre-trained DBoW3 Bag-of-Words vocabulary).
- `KeyFrame.h` (The objects being stored in the database).

---

## Block-by-Block Breakdown

### 1. The Inverted File Index

```cpp
// Associated vocabulary
const ORBVocabulary* mpVoc;

// Inverted file
std::vector<list<KeyFrame*> > mvInvertedFile;
```
**Explanation:** 
- `mpVoc` is the massive visual vocabulary (e.g., 1 million "visual words") loaded at system startup.
- `mvInvertedFile` is the core mechanism of the database. It is an array where the index corresponds to a specific visual word ID (from $0$ to $1,000,000$).
- If visual word ID `#42` is present in `KeyFrame A` and `KeyFrame B`, then `mvInvertedFile[42]` will contain a list of pointers to `[KeyFrame A, KeyFrame B]`.
- Instead of asking "What words are in this KeyFrame?", the system asks "Which KeyFrames contain this specific word?". This inverted lookup is what makes real-time image retrieval possible.

### 2. Database Maintenance

```cpp
void add(KeyFrame* pKF);
void erase(KeyFrame* pKF);
void clear();
void clearMap(Map* pMap);
```
**Explanation:** 
- Whenever a new `KeyFrame` is created and its BoW vector is computed, it must be added to the database. The `add` function loops through the visual words present in the KeyFrame and appends the KeyFrame's pointer to the corresponding lists in `mvInvertedFile`.
- `erase` removes the KeyFrame from all lists. This happens during Bundle Adjustment if a KeyFrame is deemed redundant (e.g., the drone was hovering in place, generating 10 identical KeyFrames; 9 are erased).

### 3. Querying the Database (Loop & Merge Detection)

```cpp
void DetectCandidates(KeyFrame* pKF, float minScore, vector<KeyFrame*>& vpLoopCand, vector<KeyFrame*>& vpMergeCand);
void DetectBestCandidates(KeyFrame *pKF, vector<KeyFrame*> &vpLoopCand, vector<KeyFrame*> &vpMergeCand, int nMinWords);
```
**Explanation:** 
- When the `LoopClosing` thread runs, it passes the latest `KeyFrame` into these functions.
- The functions extract the BoW words from the query `KeyFrame` and instantly fetch all historical KeyFrames that share those words via the `mvInvertedFile`.
- They compute a similarity score based on the TF-IDF (Term Frequency-Inverse Document Frequency) weights of the shared words. 
- **Loops vs. Merges:** 
  - If a high-scoring candidate KeyFrame belongs to the *same* map as the query KeyFrame, it is added to `vpLoopCand` (Triggering a Loop Closure).
  - If the candidate belongs to a *different* map (from a previous session or a period where tracking was lost and a new map was started), it is added to `vpMergeCand` (Triggering a Map Merge).

### 4. Relocalization

```cpp
std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F, Map* pMap);
```
**Explanation:** 
- If the camera moves too fast or is temporarily covered, the `Tracking` thread enters the `LOST` state.
- In this state, it stops building the map and purely queries the `KeyFrameDatabase` with the live video feed (`Frame* F`) as fast as possible, trying to find a high-scoring match to a known location in the map. Once found, tracking resumes seamlessly.
