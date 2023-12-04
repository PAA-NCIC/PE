#include<iostream>
#include<x86intrin.h>
#include<stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <asm/unistd.h>
#include <errno.h>
#include <stdint.h>
#include <inttypes.h>
#include <iomanip>

using namespace std;

//#define __AVX__
//read reference cycles
//regardless of turbo, power-saving, or clock-stopped idle
uint64_t rdtsc(){
    return __rdtsc();
}
//get time
double get_time(struct timespec *start,
  struct timespec *end)
{
  return end->tv_sec - start->tv_sec +
    (end->tv_nsec - start->tv_nsec) * 1e-9;
}

const int n = 1000000;
const int M = 1000;

#if __AVX512F__
  const int vwidth = 64;
#elif __AVX__
  const int vwidth = 32;
#endif
  typedef float floatv __attribute__((vector_size(vwidth), aligned(4)));

static long perf_event_open(struct perf_event_attr *hw_event, 
  pid_t pid, int cpu, int group_fd, unsigned long flags) {
  int ret;
  ret = syscall(SYS_perf_event_open, hw_event, pid, cpu,
                group_fd, flags);
  return ret;
}

//jb = 8, for example
template <unsigned int jb>
void axpy_simd(floatv *X, floatv a, floatv c, long m){
  for(uint32_t j = 0; j < m; j += jb) {
    for(uint64_t i = 0; i < n; i++) {
      #pragma unroll jb 
      for(uint32_t jj = j; jj < j + jb; jj++) {
        X[jj] = a * X[jj] + c;
      }
    }
  }
}

template void axpy_simd<1>(floatv*, floatv, floatv, long);
template void axpy_simd<2>(floatv*, floatv, floatv, long);
template void axpy_simd<3>(floatv*, floatv, floatv, long);
template void axpy_simd<4>(floatv*, floatv, floatv, long);
template void axpy_simd<5>(floatv*, floatv, floatv, long);
template void axpy_simd<6>(floatv*, floatv, floatv, long);
template void axpy_simd<7>(floatv*, floatv, floatv, long);
template void axpy_simd<8>(floatv*, floatv, floatv, long);
template void axpy_simd<9>(floatv*, floatv, floatv, long);
template void axpy_simd<10>(floatv*, floatv, floatv, long);
template void axpy_simd<11>(floatv*, floatv, floatv, long);
template void axpy_simd<12>(floatv*, floatv, floatv, long);
template void axpy_simd<13>(floatv*, floatv, floatv, long);
template void axpy_simd<14>(floatv*, floatv, floatv, long);
template void axpy_simd<15>(floatv*, floatv, floatv, long);
template void axpy_simd<16>(floatv*, floatv, floatv, long);
void (*kernels[16])(floatv *, floatv, floatv, long)= {
  axpy_simd<1>,
  axpy_simd<2>,
  axpy_simd<3>,
  axpy_simd<4>,
  axpy_simd<5>,
  axpy_simd<6>,
  axpy_simd<7>,
  axpy_simd<8>,
  axpy_simd<9>,
  axpy_simd<10>,
  axpy_simd<11>,
  axpy_simd<12>,
  axpy_simd<13>,
  axpy_simd<14>,
  axpy_simd<15>,
  axpy_simd<16>
};

//jb = 8, for example
template <unsigned int jb>
void axpy_simd_register(floatv *X, floatv a, floatv c, long m){
  __m512 tmp[jb];
  for(uint32_t j = 0; j < m; j += jb) {
    //load into register
    #pragma unroll jb 
    for(uint32_t jj = 0; jj < jb; jj++) {
      tmp[jj] = X[jj + j];
    }
    //compute
    for(uint64_t i = 0; i < n; i++) {
      #pragma unroll jb 
      for(uint32_t jj = 0; jj < jb; jj++) {
        tmp[jj] = a * tmp[jj] + c;
      }
    }
    //write back
    for(uint32_t jj = 0; jj < jb; jj++) {
      X[jj + j] = tmp[jj];
    }
  }
}

template void axpy_simd_register<1>(floatv*, floatv, floatv, long);
template void axpy_simd_register<2>(floatv*, floatv, floatv, long);
template void axpy_simd_register<3>(floatv*, floatv, floatv, long);
template void axpy_simd_register<4>(floatv*, floatv, floatv, long);
template void axpy_simd_register<5>(floatv*, floatv, floatv, long);
template void axpy_simd_register<6>(floatv*, floatv, floatv, long);
template void axpy_simd_register<7>(floatv*, floatv, floatv, long);
template void axpy_simd_register<8>(floatv*, floatv, floatv, long);
template void axpy_simd_register<9>(floatv*, floatv, floatv, long);
template void axpy_simd_register<10>(floatv*, floatv, floatv, long);
template void axpy_simd_register<11>(floatv*, floatv, floatv, long);
template void axpy_simd_register<12>(floatv*, floatv, floatv, long);
template void axpy_simd_register<13>(floatv*, floatv, floatv, long);
template void axpy_simd_register<14>(floatv*, floatv, floatv, long);
template void axpy_simd_register<15>(floatv*, floatv, floatv, long);
template void axpy_simd_register<16>(floatv*, floatv, floatv, long);

void (*kernels_unroll[16])(floatv *, floatv, floatv, long)= {
  axpy_simd_register<1>,
  axpy_simd_register<2>,
  axpy_simd_register<3>,
  axpy_simd_register<4>,
  axpy_simd_register<5>,
  axpy_simd_register<6>,
  axpy_simd_register<7>,
  axpy_simd_register<8>,
  axpy_simd_register<9>,
  axpy_simd_register<10>,
  axpy_simd_register<11>,
  axpy_simd_register<12>,
  axpy_simd_register<13>,
  axpy_simd_register<14>,
  axpy_simd_register<15>,
  axpy_simd_register<16>
};
int main() 
{
  struct perf_event_attr pe;
  int fd;
  memset(&pe, 0, sizeof(pe));
  pe.type = PERF_TYPE_HARDWARE;
  pe.size = sizeof(pe);
  pe.config = PERF_COUNT_HW_CPU_CYCLES;
  pe.disabled = 1;
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;
  fd = perf_event_open(&pe, 0, -1, -1, 0);
  if (fd == -1) {
    fprintf(stderr, "Error opening leader %llx\n", pe.config);
    exit(EXIT_FAILURE);
  }

  const int L = sizeof(floatv) / sizeof(float);
  cout << "SIMD width: " << L << endl;
  floatv a, c;
  floatv *x = (floatv *)aligned_alloc(512, M * sizeof(floatv));
  for(int i = 0; i < L; i++)
    a[i] = c[i]= 1.0;
  for(int i = 0; i < M; i++)
    x[i] = a;

  uint64_t start_cycle, end_cycle;
  struct timespec start, end;
  //header
  cout << "compiler may do a bad job..." << endl;
  cout << setw(20) << "chains," << "\t" << setw(20) << "cycles/iter," \
  << "\t" << "flops/cycle" << endl;
  
  //the same code as in PPT, but our compiler
  //did not optimize as we expected
  for(int i = 0; i < 16; i++ ) {
    kernels[i](x, a, c, 1024);
    //time
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    //cpu ref clocks
    start_cycle = rdtsc();
    //cpu core clocks
    ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    //to ensure that m is a multiple of jb
    //though m is slightly different, but should
    //have no impact on the performance
    int m = M / (i+1) * (i+1);
    kernels[i](x, a, c, m); 
    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    end_cycle = rdtsc();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    double used_time = get_time(&start, &end);
    uint64_t used_cycles= end_cycle - start_cycle;
    double flops = 2.0 * L * n * m;
    uint64_t cpu_clocks;
    if(read(fd, &cpu_clocks, sizeof(cpu_clocks)) == -1) {
       cout << "read cpu clocks error" << endl;
    };
    cout << setw(20) << (i+1) << ",\t" << setw(20) << 1.0 * cpu_clocks / n / (m / (i + 1)) \
    << ",\t" << flops / cpu_clocks << endl;
  }  
  //register optimize
  cout << "we can help compiler do a better job..."
  cout << setw(20) << "chains," << "\t" << setw(20) << "cycles/iter," \
  << "\t" << "flops/cycle" << endl;
  for(int i = 0; i < 16; i++ ) {
    kernels[i](x, a, c, 1024);
    //time
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    //cpu ref clocks
    start_cycle = rdtsc();
    //cpu core clocks
    ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    //to ensure that m is a multiple of jb
    //though m is slightly different, but should
    //have no impact on the performance
    int m = M / (i+1) * (i+1);
    kernels_unroll[i](x, a, c, m); 
    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    end_cycle = rdtsc();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    double used_time = get_time(&start, &end);
    uint64_t used_cycles= end_cycle - start_cycle;
    double flops = 2.0 * L * n * m;
    uint64_t cpu_clocks;
    if(read(fd, &cpu_clocks, sizeof(cpu_clocks)) == -1) {
       cout << "read cpu clocks error" << endl;
    };
    cout << setw(20) << (i+1) << ",\t" << setw(20) << 1.0 * cpu_clocks / n / (m / (i + 1)) \
    << ",\t" << flops / cpu_clocks << endl;
  }  
  //in case that compiler optimized
  return (int)x[0][0];
}
