# Documentation: `Timestamp.cpp`

## High-Level Overview
The `Timestamp.cpp` file provides a lightweight, cross-platform utility class for managing high-precision time values.
In SLAM systems, particularly Visual-Inertial SLAM, time synchronization is mission-critical. A camera might operate at 30Hz, while an IMU operates at 200Hz. If the timestamp of a camera frame is off by even a few milliseconds relative to the IMU data, the mathematical integration will fail, and the drone will crash. This class ensures time is handled consistently, robustly, and accurately across different operating systems (Linux, Windows).

**Primary Dependencies:**
- `Timestamp.h`
- `<sys/time.h>` (Linux microsecond resolution)
- `<sys/timeb.h>` (Windows millisecond resolution fallback)

---

## Block-by-Block Breakdown

### 1. Cross-Platform Macros and Includes

```cpp
#ifdef WIN32
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif
```
**Explanation:** 
- The class is designed to compile cleanly on both POSIX (Linux/macOS) and Windows systems.
- POSIX systems natively support microsecond precision using `gettimeofday`.
- Older Windows systems rely on `_ftime32_s`, which only provides millisecond (1/1000th of a second) precision.

### 2. Time Capture and Internal Representation

```cpp
void Timestamp::setToCurrentTime(){ ... }
void Timestamp::setTime(double s)
```
**Explanation:** 
- The class internally stores time as two `unsigned long` integers: `m_secs` (seconds since the Epoch) and `m_usecs` (microseconds).
- Storing time as a single `double` float can lead to floating-point truncation errors when dealing with very large Unix epoch timestamps combined with microsecond-level precision. Splitting it into two integers preserves perfect precision.
- `setToCurrentTime` queries the OS for the current wall-clock time and populates these variables.

### 3. String Parsing

```cpp
void Timestamp::setTime(const string &stime)
```
**Explanation:** 
- When reading datasets like EuroC, KITTI, or TUM, timestamps are often provided as massive string numbers (e.g., `"1403636579.763555"`).
- This function explicitly parses the string, searching for the decimal point. It assigns the left side to `m_secs` and scales the right side appropriately to microseconds. This prevents the loss of precision that would occur if you simply used `stod()` (string to double) on a 16-digit timestamp.

### 4. Mathematical Operators

```cpp
Timestamp Timestamp::plus(unsigned long secs, unsigned long usecs) const
Timestamp Timestamp::minus(unsigned long secs, unsigned long usecs) const
bool Timestamp::operator> (const Timestamp &t) const
```
**Explanation:** 
- The file overloads all standard mathematical operators (`+`, `-`, `+=`, `-=`) and comparison operators (`<`, `>`, `==`).
- The addition and subtraction logic carefully handles the carry-over for microseconds (since 1,000,000 microseconds = 1 second). If `m_usecs` overflows $10^6$, it increments `m_secs` and subtracts $10^6$ from `m_usecs`.
- The comparison operators prioritize checking `m_secs` first, only checking `m_usecs` if the seconds are identical.

### 5. Formatting and Display

```cpp
string Timestamp::Format(bool machine_friendly) const
string Timestamp::Format(double s)
```
**Explanation:** 
- These functions are used for logging and debug output.
- `Format(bool)` converts the Epoch time back into a human-readable calendar date using standard C library functions (`localtime_r`, `strftime`).
- `Format(double)` breaks down a raw duration (in seconds) into Days, Hours, Minutes, Seconds, and Milliseconds. This is particularly useful for tracking how long a SLAM trajectory has been running.
