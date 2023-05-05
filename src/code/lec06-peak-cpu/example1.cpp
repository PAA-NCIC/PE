#include<iostream>
#include<x86intrin.h>
#include<stdio.h>
using namespace std;

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

const int n = 100000000;
int main() 
{
#if __AVX512F__
  const int vwidth = 64;
#elif __AVX__
  const int vwidth = 32;
#endif
  const int valign = sizeof(float);
  typedef float floatv __attribute__((vector_size(vwidth), aligned(valign)));
  const int L = sizeof(floatv);
  cout << "floatv: " << L << endl;
  floatv a, x, c;
  a[0] = x[0] = c[0] = 1.0;
  a[1] = x[1] = c[1] = 1.0;
  a[2] = x[2] = c[2] = 1.0;
  a[3] = x[3] = c[3] = 1.0;
  uint64_t start_cycle, end_cycle;
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  start_cycle = rdtsc();
  for (int i = 0; i < n; i++) {
    x = a * x + c;
  }
  end_cycle = rdtsc();
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double used_time = get_time(&start, &end);
  uint64_t used_cycles = end_cycle - start_cycle;
  uint64_t flops = 2 * vwidth / 4 * n;
  cout << "Flops: " << flops << endl;
  cout << "cpu ref cycles: " << used_cycles << endl;
  cout << "flops per ref cycle: " << 1.0 * flops / used_cycles << endl;
  cout << "Achieve " << flops / used_time << " flops" << endl;
  return 0;
}
