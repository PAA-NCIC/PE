#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdbool.h>
#include <omp.h>
#include <sys/types.h>
#include <sys/time.h>

#define L1_CACHE_SIZE (48 * 1024)
#define L2_CACHE_SIZE (1280 * 1024)
#define L3_CACHE_SIZE (55296 * 1024)

#define min(x, y) ((x) < (y) ? (x) : (y))
#define max(x, y) ((x) < (y) ? (y) : (x))

inline double dtime(){
    struct timeval t;
    gettimeofday(&t, 0);
    return t.tv_sec + 1e-6 * t.tv_usec;
}

static int env_get_int(const char* name, int default_value){
    char* tmp = getenv(name);
    if(tmp == NULL){
        return default_value;
    }
    return atoi(tmp);
}

static uint64_t env_get_uint64(const char* name, uint64_t default_value){
    char* tmp = getenv(name);
    if(tmp == NULL){
        return default_value;
    }
    return atol(tmp);
}


static const char* env_get_string(const char* name, const char* default_value){
    char* tmp = getenv(name);
    if(tmp == NULL){
        return default_value;
    }
    return tmp;
}

static inline uint64_t rdtsc(void)
{
    uint64_t msr;
    __asm__ volatile ( "rdtsc\n\t"    // Returns the time in EDX:EAX.
               "shl $32, %%rdx\n\t"  // Shift the upper bits left.
               "or %%rdx, %0"        // 'Or' in the lower bits.
               : "=a" (msr)
               :
               : "rdx");
    return msr;
}
