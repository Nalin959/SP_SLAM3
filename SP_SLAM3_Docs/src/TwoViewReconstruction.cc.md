# Documentation: `TwoViewReconstruction.cc`

## High-Level Overview
The `TwoViewReconstruction.cc` file solves one of the most classic and difficult problems in computer vision: **Monocular Initialization**.
When a single camera turns on, it has no depth perception. To start SLAM, the camera must move slightly to capture a second frame. By tracking features between Frame 1 and Frame 2, this module mathematically reconstructs the 3D structure of the scene and the relative motion (Translation and Rotation) between the two frames, completely from scratch.

This solver operates under the assumption that the scene can either be planar (best modeled by a Homography) or non-planar with significant depth variation (best modeled by a Fundamental Matrix). It computes both in parallel and uses a heuristic to decide which model explains the data better.

**Primary Dependencies:**
- `TwoViewReconstruction.h`
- `Random.h` (For RANSAC sampling).
- OpenCV (`cv::SVDecomp`, `cv::determinant`) for the heavy linear algebra.

---

## Block-by-Block Breakdown

### 1. The Main Reconstructor (`Reconstruct`)

```cpp
bool TwoViewReconstruction::Reconstruct(...)
```
**Explanation:** 
- Takes two sets of keypoints (`vKeys1`, `vKeys2`) and the matches between them (`vMatches12`).
- **RANSAC Pre-computation:** It pre-generates the random indices for hundreds of RANSAC iterations upfront (`mvSets`). In each iteration, it selects exactly 8 points (the theoretical minimum required to compute a Fundamental Matrix using the 8-point algorithm).
- **Parallel Computing:** It launches two separate C++ threads (`std::thread`):
  1. `FindHomography` (computes the $3 \times 3$ matrix `H`).
  2. `FindFundamental` (computes the $3 \times 3$ matrix `F`).
- **Model Selection:** Once both threads finish, it calculates a ratio `RH = SH / (SH + SF)` where `SH` and `SF` are the inlier scores of the best Homography and Fundamental matrix respectively.
  - If `RH > 0.50`, the scene is heavily planar (e.g., looking straight down at a flat floor) or the camera didn't translate enough. It initializes using `ReconstructH`.
  - Otherwise, the scene has good 3D structure. It initializes using `ReconstructF`.

### 2. Finding the Matrices (RANSAC Threads)

```cpp
void TwoViewReconstruction::FindHomography(...)
void TwoViewReconstruction::FindFundamental(...)
```
**Explanation:** 
- Both threads follow identical RANSAC logic.
- **Normalization (`Normalize`):** Before running the algorithms, it translates the keypoints so their centroid is at $(0,0)$ and scales them so their average distance from the origin is $\sqrt{2}$. *This is Hartley's normalization, absolutely critical for the numerical stability of the Singular Value Decomposition (SVD).*
- Inside the loop, it takes the 8 normalized points, computes the matrix (`ComputeH21` or `ComputeF21`), denormalizes the resulting matrix, and calculates a reprojection error score across *all* points (`CheckHomography` or `CheckFundamental`).
- It keeps the matrix that achieves the highest score.

### 3. The 8-Point Algorithm (`ComputeF21`)

```cpp
cv::Mat TwoViewReconstruction::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
```
**Explanation:** 
- Implements Longuet-Higgins' 8-point algorithm.
- For every match $(u_1, v_1) \leftrightarrow (u_2, v_2)$, it sets up the epipolar constraint equation: $x_2^T F x_1 = 0$.
- This forms an $8 \times 9$ matrix `A`.
- `cv::SVDecomp(A)`: The Fundamental matrix is the right-singular vector associated with the smallest singular value (the last column of `Vt`, or the 9th row in OpenCV's output).
- **Rank-2 Constraint:** A valid Fundamental Matrix must have a determinant of 0 (Rank 2). The raw matrix from the first SVD is Rank 3 due to noise. So, it performs a *second* SVD on `Fpre`, explicitly sets the smallest singular value `w.at<float>(2)` to 0, and reassembles the matrix. This guarantees the Epipolar lines perfectly intersect at the epipoles.

### 4. Decomposition and 3D Triangulation

```cpp
bool TwoViewReconstruction::ReconstructF(...)
```
**Explanation:** 
- Once `F21` is selected as the winning model, it must extract the physical camera rotation and translation.
- It computes the Essential Matrix $E = K^T F K$.
- `DecomposeE`: SVD decomposition of the Essential Matrix yields exactly 4 mathematical solutions (four possible combinations of Rotation $R$ and Translation $t$).
- **The Chierality Check (`CheckRT`):** Only one of these 4 hypotheses is physically possible (the one where the triangulated 3D points appear *in front* of both cameras). 
- It triangulates the points for all 4 hypotheses. The hypothesis that yields the most points with positive depth ($Z > 0$) is chosen as the true trajectory.

### 5. Triangulation Math

```cpp
void TwoViewReconstruction::Triangulate(...)
```
**Explanation:** 
- Implements the Direct Linear Transform (DLT) for triangulation.
- Given the projection matrices $P_1$ and $P_2$, it sets up a system of linear equations $AX = 0$.
- Solving this via SVD (`cv::SVD::compute`) yields the 3D point $X$ in homogeneous coordinates $(x, y, z, w)$. It divides by $w$ to get the final 3D Euclidean coordinates.
