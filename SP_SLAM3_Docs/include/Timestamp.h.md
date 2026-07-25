# Documentation: `Timestamp.h`

## High-Level Overview
The `Timestamp.h` file is a utility class created by Dorian Galvez-Lopez, providing a clean, cross-platform wrapper for handling high-precision timing.
In a SLAM system, microsecond-level timing is critical for two main reasons:
1. **Sensor Synchronization:** When fusing data from a camera running at 30Hz and an IMU running at 200Hz, the system must precisely interpolate the IMU data that occurred exactly between two camera exposures. An error of just 5 milliseconds can ruin the inertial optimization.
2. **Performance Profiling:** Developers need to measure exactly how many milliseconds the `Tracking` or `LoopClosing` threads take to execute to ensure the system is running in real-time.

**Primary Dependencies:**
- Standard C++ Libraries (`<iostream>`, `<string>`).

---

## Block-by-Block Breakdown

### 1. The Core Data Structure

```cpp
class Timestamp
{
protected:
    unsigned long m_secs;  // seconds
    unsigned long m_usecs; // microseconds
};
```
**Explanation:** 
- The class internally stores time as two separate variables: whole seconds (usually elapsed since the UNIX Epoch in 1970) and remainder microseconds. 
- Splitting them up prevents the catastrophic floating-point precision loss that would occur if you tried to store UNIX timestamps down to the microsecond in a single `double` (a standard 64-bit IEEE double cannot accurately represent precision that fine for numbers that large).

### 2. Time Capture and Conversion

```cpp
void setToCurrentTime();
void setTime(double s);
void setTime(const string &stime);
double getFloatTime() const;
string getStringTime() const;
```
**Explanation:** 
- `setToCurrentTime`: Grabs the current system clock time (using OS-specific high-resolution timers under the hood in the `.cpp` file). Used heavily for profiling.
- `setTime(double)` / `setTime(string)`: SLAM datasets (like EuRoC or KITTI) provide image timestamps in text files. These functions parse those dataset timestamps into the `m_secs` / `m_usecs` structure.
- `getFloatTime()`: Converts the internal structure back into a standard `double`. Useful for quick, low-precision math.

### 3. Operator Overloading (Time Math)

```cpp
double operator- (const Timestamp &t) const;
Timestamp& operator+= (double s);
bool operator> (const Timestamp &t) const;
bool operator== (const Timestamp &t) const;
```
**Explanation:** 
- `operator-`: The most used function. `TimeB - TimeA` returns the exact elapsed duration between two events in floating-point seconds.
- `operator>` / `operator<`: Used extensively by the IMU preintegration logic. The system keeps a buffer of hundreds of IMU readings. When a new camera frame arrives, it uses these operators to iterate through the buffer and extract only the IMU readings whose timestamps fall between `Frame A` and `Frame B`.

### 4. Human-Readable Formatting

```cpp
string Format(bool machine_friendly = false) const;
static string Format(double s);
```
**Explanation:** 
- Converts internal time into human-readable strings.
- `Format(false)` might output "12d 04:30:15".
- `Format(true)` outputs `yyyymmdd_hhmmss`, which is extremely useful when the SLAM system needs to automatically generate uniquely named output files (e.g., `trajectory_20231014_153000.txt`) without fear of overwriting previous flight logs.
