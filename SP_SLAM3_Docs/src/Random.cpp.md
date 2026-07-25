# Documentation: `Random.cpp`

## High-Level Overview
The `Random.cpp` file contains a set of simple pseudo-random number generator utilities. 
While it seems trivial, generating random numbers is absolutely critical for the RANSAC (Random Sample Consensus) algorithms used extensively throughout ORB-SLAM3 (in PnP, Sim3 matching, and Fundamental matrix calculation). RANSAC relies on repeatedly drawing minimal, random sets of data points to find an outlier-free consensus set.

**Primary Dependencies:**
- `Random.h`, `Timestamp.h`
- Standard C libraries (`<cstdlib>`) for `rand()` and `srand()`.

---

## Block-by-Block Breakdown

### 1. Seeding the RNG

```cpp
bool DUtils::Random::m_already_seeded = false;

void DUtils::Random::SeedRand(){
    Timestamp time;
    time.setToCurrentTime();
    srand((unsigned)time.getFloatTime()); 
}

void DUtils::Random::SeedRandOnce()
{
  if(!m_already_seeded) {
    DUtils::Random::SeedRand();
    m_already_seeded = true;
  }
}
```
**Explanation:** 
- RANSAC algorithms need an initial seed to ensure that every run explores a different subset of the data space.
- `SeedRand()` seeds the standard C `srand()` function using the current system microsecond time (`Timestamp::getFloatTime()`).
- `SeedRandOnce()` ensures that the global RNG is seeded exactly once during the program's lifecycle, preventing the accidental reset of the random sequence if called repeatedly in a tight loop. Note that `m_already_seeded` is a static class member.

### 2. Random Integer Generation

```cpp
int DUtils::Random::RandomInt(int min, int max){
    int d = max - min + 1;
    return int(((double)rand()/((double)RAND_MAX + 1.0)) * d) + min;
}
```
**Explanation:** 
- Generates a random integer in the range `[min, max]`. 
- By casting to `double` and dividing by `RAND_MAX + 1.0`, it avoids the slight modulo bias that occurs when simply doing `rand() % d`.

### 3. Unrepeated Randomizer

```cpp
DUtils::Random::UnrepeatedRandomizer::UnrepeatedRandomizer(int min, int max)
{
    // ...
    createValues();
}

void DUtils::Random::UnrepeatedRandomizer::createValues()
{
    int n = m_max - m_min + 1;
    m_values.resize(n);
    for(int i = 0; i < n; ++i) m_values[i] = m_min + i;
}
```
**Explanation:** 
- When RANSAC needs to pick, for example, 8 points out of 100, it cannot pick the same point twice in the same hypothesis (which would result in degenerate mathematics).
- The `UnrepeatedRandomizer` class guarantees that it will draw numbers without replacement. 
- On initialization, `createValues` populates a vector `m_values` with every integer in the range sequentially (e.g., `[0, 1, 2, ..., 99]`).

```cpp
int DUtils::Random::UnrepeatedRandomizer::get()
{
    if(empty()) createValues();
    DUtils::Random::SeedRandOnce();
    
    int k = DUtils::Random::RandomInt(0, m_values.size()-1);
    int ret = m_values[k];
    
    // Fast O(1) removal without shifting the array
    m_values[k] = m_values.back();
    m_values.pop_back();
    
    return ret;
}
```
**Explanation:** 
- When `get()` is called, it picks a random index `k` from the currently available pool.
- It saves the value to return.
- **Optimization:** To remove the value from the pool, it does not use `std::vector::erase()` (which would be $O(N)$ because it shifts all subsequent elements). Instead, it swaps the chosen element with the very last element in the array (`m_values[k] = m_values.back()`) and then pops the back. This achieves $O(1)$ removal, which is highly efficient for the thousands of RANSAC iterations executed per second.
