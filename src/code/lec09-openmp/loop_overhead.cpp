#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
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
#include <time.h>

using namespace std;

#define N 512
#define LOOP 100000000

//get time
double get_time(struct timespec *start,
  struct timespec *end)
{
  return end->tv_sec - start->tv_sec +
    (end->tv_nsec - start->tv_nsec) * 1e-9;
}

int main(int argc, char* argv[]){
    float A[N], B[N], C[N], D[N];
    for(int i = 0; i < N; i++) {
      A[i] = B[i] = C[i] = D[i];
    }
   
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for(int iter = 1; iter < LOOP; iter++) {
      #pragma omp parallel 
      {
      #pragma omp parallel for 
        for(int i = 0; i < N; i++) {
          A[i] = B[i] + C[i] * D[i];
        }
      }
    }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double used_time = get_time(&start, &end);
  double flops = 1.0 * N * 2 * LOOP;
  cout << "achieved flops:" << flops / used_time << endl;
    return 0;
}

