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

struct read_format {
  uint64_t nr;
  struct {
    uint64_t value;
    uint64_t id;
  } values[];
};

int main() 
{
  struct perf_event_attr pea;
  int fd1;
  uint64_t id1;
  uint64_t cpu_clocks;
  char buf[4096];
  struct read_format* rf = (struct read_format*) buf;
  int i;

  memset(&pea, 0, sizeof(struct perf_event_attr));
  pea.type = PERF_TYPE_HARDWARE;
  pea.size = sizeof(struct perf_event_attr);
  pea.config = PERF_COUNT_HW_CPU_CYCLES;
  pea.disabled = 1;
  pea.exclude_kernel = 1;
  pea.exclude_hv = 1;
  pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
  fd1 = syscall(__NR_perf_event_open, &pea, 0, -1, -1, 0);
  ioctl(fd1, PERF_EVENT_IOC_ID, &id1);

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
  a[0] = x[0] = c[0] = 1.0;
  a[1] = x[1] = c[1] = 1.0;
  a[2] = x[2] = c[2] = 1.0;
  a[3] = x[3] = c[3] = 1.0;
  uint64_t start_cycle, end_cycle;
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  start_cycle = rdtsc();
  ioctl(fd1, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
  ioctl(fd1, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
  for (int i = 0; i < n; i++) {
    x = a * x + c;
  }
  ioctl(fd1, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
  end_cycle = rdtsc();
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double used_time = get_time(&start, &end);
  uint64_t used_cycles = end_cycle - start_cycle;
  uint64_t flops = 2 * L * n;
  read(fd1, buf, sizeof(buf));
  for (i = 0; i < rf->nr; i++) {
    if (rf->values[i].id == id1) {
      cpu_clocks = rf->values[i].value;
    }
  }
  cout << "Flops: " << flops << endl;
  cout << "cpu ref cycles: " << used_cycles << endl;
  cout << "flops per ref cycle: " << 1.0 * flops / used_cycles << endl;
  cout << "cpu clocks: " << cpu_clocks << endl;
  cout << "flops per cycle: " << 1.0 * flops / cpu_clocks << endl;
  cout << "Ghz " << 1.0 * used_cycles /cpu_clocks << endl;
  cout << "Achieve " << flops / used_time << " flops" << endl;
  return (int)x[0];
}
