# Documentation: `Random.h`

## High-Level Overview
The `Random.h` file is a small utility class developed by Dorian Galvez-Lopez (the original author of DBoW2). It provides a clean, C++ wrapper around the standard C `<cstdlib>` random number generators.
In a SLAM system, pseudo-random number generation is surprisingly critical. It is used constantly by RANSAC loops (to randomly select subsets of point matches) and by Bag-of-Words clustering algorithms (e.g., K-Means++ initialization).

**Primary Dependencies:**
- `<cstdlib>` (For `rand()` and `srand()`).

---

## Block-by-Block Breakdown

### 1. The Global Randomizer

```cpp
static void SeedRand();
static void SeedRandOnce();
template <class T> static T RandomValue();
template <class T> static T RandomValue(T min, T max);
static int RandomInt(int min, int max);
```
**Explanation:** 
- `SeedRand()`: Seeds the global random number generator with the current time (usually `time(NULL)`). This ensures every run of the SLAM system behaves slightly differently.
- **Reproducibility Note:** For debugging, developers will often pass a fixed seed (e.g., `SeedRand(0)`) so that RANSAC selects the exact same points every time they run the code, ensuring deterministic behavior.
- `RandomValue` / `RandomInt`: Clean wrappers that scale the output of `rand()` (which outputs an integer from $0$ to `RAND_MAX`) into a desired floating-point or integer range $[min, max]$.

### 2. Gaussian Noise Generator

```cpp
template <class T> static T RandomGaussianValue(T mean, T sigma)
```
**Explanation:** 
- The standard `rand()` function produces a *Uniform* distribution (every number is equally likely).
- In robotics, sensor noise (like camera pixel noise or IMU vibration) is almost always modeled as a *Gaussian* (Normal) distribution (a bell curve).
- This function uses the **Box-Muller transform** to convert two uniformly distributed random variables into a normally distributed random variable with the specified `mean` and standard deviation (`sigma`). This is highly useful for writing simulation code that generates fake, noisy sensor data.

### 3. The Unrepeated Randomizer

```cpp
class Random::UnrepeatedRandomizer
{
  int get();
  void reset();
  std::vector<int> m_values;
}
```
**Explanation:** 
- In RANSAC, if you want to select 8 random 2D-3D matches, you must ensure you don't pick the *exact same* match twice.
- If you just call `RandomInt(0, 100)` 8 times, there is a statistical chance of collision.
- `UnrepeatedRandomizer` solves this. When created with `min` and `max` (e.g., $0$ and $100$), it builds an internal vector (`m_values`) containing every single number in that range. When `get()` is called, it randomly selects an index, returns the number, and immediately deletes it from `m_values`. This guarantees zero repetitions, analogous to dealing cards from a deck.
