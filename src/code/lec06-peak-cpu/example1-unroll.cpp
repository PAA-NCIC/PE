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

static long perf_event_open(struct perf_event_attr *hw_event, 
  pid_t pid, int cpu, int group_fd, unsigned long flags) {
  int ret;
  ret = syscall(SYS_perf_event_open, hw_event, pid, cpu,
                group_fd, flags);
  return ret;
}

int main() 
{
  struct perf_event_attr pe;
  int fd;
  uint64_t cpu_clocks;
  char buf[4096];
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


#if __AVX512F__
  const int vwidth = 64;
#elif __AVX__
  const int vwidth = 32;
#endif
  const int valign = sizeof(float);
  typedef float floatv __attribute__((vector_size(vwidth), aligned(valign)));
  const int L = sizeof(floatv) / sizeof(float);
  cout << "SIMD width: " << L << endl;
  floatv a, x, c;
  for(int i = 0; i < L; i++)
    a[i] = x[i] = c[i] = 1.0;
 
  uint64_t start_cycle, end_cycle;
  struct timespec start, end;
  //warm up
  for (int i = 0; i < n; i++) {
    x = a * x + c;
  }
  //time
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  //cpu ref clocks
  start_cycle = rdtsc();
  //cpu core clocks
  ioctl(fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);

asm volatile ("# axpy simd begin");
  for (int i = 0; i < n / 8; i++) {
    #pragma unroll 8
    for(int j = 0; j < 8; j++) {
      x = a * x + c;
    }
  }
asm volatile ("# axpy simd end");
  ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
  end_cycle = rdtsc();
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double used_time = get_time(&start, &end);
  uint64_t used_cycles = end_cycle - start_cycle;
  double flops = 2.0 * L * n;
  if(read(fd, &cpu_clocks, sizeof(cpu_clocks)) == -1) {
    cout << "read cpu clocks error" << endl;
  }

  //print out
  cout << "Flops:                " << flops << endl;
  cout << "cpu ref cycles:       " << used_cycles << endl;
  cout << "flops per ref cycle:  " << 1.0 * flops / used_cycles << endl;
  cout << "cpu core clocks:      " << cpu_clocks << endl;
  cout << "flops per core cycle: " << 1.0 * flops / cpu_clocks << endl;
  cout << "iter per core cycle:  " << 1.0 * cpu_clocks / n << endl;
  //cout << "Ghz " << 1.0 * used_cycles /cpu_clocks << endl;
  cout << "Achieve flops:        " << flops / used_time  << endl;
  
  //in case that compiler optimized
  return (int)x[0];
}
