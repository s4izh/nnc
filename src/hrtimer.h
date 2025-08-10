#ifndef HRTIMER_H_
#define HRTIMER_H_

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>

typedef struct {
    LARGE_INTEGER start;
    LARGE_INTEGER stop;
} stopwatch_t;

static inline LARGE_INTEGER get_timer_frequency() {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    return frequency;
}

static inline void stopwatch_start(stopwatch_t* sw) {
    QueryPerformanceCounter(&sw->start);
}

static inline void stopwatch_stop(stopwatch_t* sw) {
    QueryPerformanceCounter(&sw->stop);
}

static inline double stopwatch_get_elapsed_seconds(stopwatch_t* sw, LARGE_INTEGER frequency) {
    return (double)(sw->stop.QuadPart - sw->start.QuadPart) / frequency.QuadPart;
}

#else // For Linux, macOS, and other POSIX-compliant systems
#include <time.h>

typedef struct {
    struct timespec start;
    struct timespec stop;
} stopwatch_t;

static inline long get_timer_frequency() {
    return 1e9;
}

static inline void stopwatch_start(stopwatch_t* sw) {
    clock_gettime(CLOCK_MONOTONIC, &sw->start);
}

static inline void stopwatch_stop(stopwatch_t* sw) {
    clock_gettime(CLOCK_MONOTONIC, &sw->stop);
}

static inline double stopwatch_get_elapsed_seconds(stopwatch_t* sw, long frequency) {
    (void)frequency;
    return (sw->stop.tv_sec - sw->start.tv_sec) +
           (sw->stop.tv_nsec - sw->start.tv_nsec) / 1e9;
}

#endif // _WIN32 || _WIN64

#endif // HRTIMER_H_
