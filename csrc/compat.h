#ifndef COMPAT_H
#define COMPAT_H

/* Enable POSIX extensions (clock_gettime, posix_memalign, etc.) */
#ifndef _POSIX_C_SOURCE
#  define _POSIX_C_SOURCE 200809L
#endif

/* Platform & compiler compatibility */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* ── Compiler attributes ────────────────────────────────────────── */
#ifdef _MSC_VER
#  define FORCE_INLINE __forceinline
#  define ALIGN(n)     __declspec(align(n))
#  define LIKELY(x)    (x)
#  define UNLIKELY(x)  (x)
#  include <intrin.h>
#else
#  define FORCE_INLINE __attribute__((always_inline)) inline
#  define ALIGN(n)     __attribute__((aligned(n)))
#  define LIKELY(x)    __builtin_expect(!!(x), 1)
#  define UNLIKELY(x)  __builtin_expect(!!(x), 0)
#endif

/* ── SIMD detection ─────────────────────────────────────────────── */
#if defined(__AVX2__)
#  include <immintrin.h>
#  define HAVE_AVX2 1
#  define SIMD_WIDTH 8   /* 8 floats per AVX2 register */
#elif defined(__SSE4_1__)
#  include <smmintrin.h>
#  define HAVE_SSE4 1
#  define SIMD_WIDTH 4
#else
#  define SIMD_WIDTH 1
#endif

/* ── Thread portability ─────────────────────────────────────────── */
#if defined(_WIN32) || defined(_WIN64)
#  define PLATFORM_WINDOWS 1
#  include <windows.h>
#  include <process.h>
#  define PATH_SEP '\\'
#  define CLEAR_SCREEN "cls"
#else
#  define PLATFORM_POSIX 1
#  include <pthread.h>
#  include <unistd.h>
#  define PATH_SEP '/'
#  define CLEAR_SCREEN "clear"
#endif

/* ── OpenMP ─────────────────────────────────────────────────────── */
#ifdef _OPENMP
#  include <omp.h>
#  define OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(static)")
#  define OMP_PARALLEL_FOR_REDUCE(op,var) _Pragma("omp parallel for reduction(" #op ":" #var ")")
#else
#  define OMP_PARALLEL_FOR
#  define OMP_PARALLEL_FOR_REDUCE(op,var)
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_thread_num(void)  { return 0; }
#endif

/* ── Math helpers ───────────────────────────────────────────────── */
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#  define M_E  2.71828182845904523536
#endif

static FORCE_INLINE float fast_exp(float x) {
    /* Clamp to avoid overflow */
    if (x > 88.0f)  return 3.40282347e+38f;
    if (x < -88.0f) return 0.0f;
    return expf(x);
}

static FORCE_INLINE float fast_log(float x) {
    return logf(x < 1e-10f ? 1e-10f : x);
}

/* ── Memory helpers ─────────────────────────────────────────────── */
static FORCE_INLINE void* xmalloc(size_t n) {
    void* p = malloc(n);
    if (UNLIKELY(!p)) { fprintf(stderr, "OOM: malloc(%zu)\n", n); exit(1); }
    return p;
}
static FORCE_INLINE void* xcalloc(size_t n, size_t sz) {
    void* p = calloc(n, sz);
    if (UNLIKELY(!p)) { fprintf(stderr, "OOM: calloc(%zu,%zu)\n", n, sz); exit(1); }
    return p;
}
static FORCE_INLINE void* xrealloc(void* p, size_t n) {
    void* q = realloc(p, n);
    if (UNLIKELY(!q && n)) { fprintf(stderr, "OOM: realloc(%zu)\n", n); exit(1); }
    return q;
}

/* Aligned allocation */
static FORCE_INLINE void* xmalloc_aligned(size_t n, size_t align) {
#ifdef PLATFORM_WINDOWS
    void* p = _aligned_malloc(n, align);
#else
    void* p = NULL;
    posix_memalign(&p, align, n);
#endif
    if (UNLIKELY(!p)) { fprintf(stderr, "OOM: aligned_malloc(%zu)\n", n); exit(1); }
    return p;
}
static FORCE_INLINE void xfree_aligned(void* p) {
#ifdef PLATFORM_WINDOWS
    _aligned_free(p);
#else
    free(p);
#endif
}

/* ── String helpers ─────────────────────────────────────────────── */
static FORCE_INLINE char* xstrdup(const char* s) {
    size_t n = strlen(s) + 1;
    char*  d = (char*)xmalloc(n);
    memcpy(d, s, n);
    return d;
}

/* ── Timing ─────────────────────────────────────────────────────── */
static FORCE_INLINE double now_sec(void) {
#ifdef PLATFORM_WINDOWS
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/* ── Min / Max ──────────────────────────────────────────────────── */
#define FG_MAX(a,b) ((a) > (b) ? (a) : (b))
#define FG_MIN(a,b) ((a) < (b) ? (a) : (b))
#define FG_CLAMP(x,lo,hi) FG_MAX((lo), FG_MIN((hi), (x)))

#endif /* COMPAT_H */
