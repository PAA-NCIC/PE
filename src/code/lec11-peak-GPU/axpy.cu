#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <x86intrin.h>
#include <stdio.h>
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

#define SIMD_TEST(block_size,C) do{ \
    double fmas = N * C;  \
    axpy_simd<C><<<1,block_size>>>(N, 1.2, d_x, 1, d_cycles);  \
    cudaDeviceSynchronize();  \
    err = cudaMemcpy(&cycles, d_cycles,  sizeof(uint32_t), cudaMemcpyDeviceToHost); \
    if (err != cudaSuccess){  \
      fprintf(stderr, "Failed to copy back cycles to CPU (error code %s)!\n", cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);  \
    }  \
    cout << "thread num per block: " << block_size << endl;\
    cout << "simd width:           " << C << endl;  \
    cout << "fmas:                 " << flops << endl;  \
    cout << "GPU cycles:           " << cycles << endl;  \
    cout << "fma per GPU cycle:    " << fmas / cycles << endl;  \
    cout << "GPU clocks per iter:  " << cycles / N  << endl;   \
} while(0)

#define N_THREADS_TEST(N,MAX,C) do {\
  SIMD_TEST(N, C);  \
  N = N + 32; \
} while(N < MAX)  
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


static long perf_event_open(struct perf_event_attr *hw_event, 
  pid_t pid, int cpu, int group_fd, unsigned long flags) {
  int ret;
  ret = syscall(SYS_perf_event_open, hw_event, pid, cpu,
                group_fd, flags);
  return ret;
}

__global__ void axpy(long n, float a, float *x, float b, uint32_t *cycles) {
  uint32_t start_time=0, stop_time=0;
  start_time = clock();
  uint32_t j = 0;
  for (int i = 0; i < n; i++) {
    x[j] = a * x[j] + b;
  }
  stop_time = clock();
  *cycles = stop_time - start_time;
  //printf("%u\n", stop_time - start_time);
}

__global__ void axpy_unroll8(long n, float a, float *x, float b, uint32_t *cycles) {
  uint32_t start_time=0, stop_time=0;
  uint32_t j = 0;
  start_time = clock();
  for (int i = 0; i < n; i+=8) {
    x[j] = x[j] + b; //x[i] = a * x[i] + b[i];
    x[j] = x[j] + b; // x[i+1] = a * x[i+1] + b[i+1];
    x[j] = x[j] + b; // x[i+2] = a * x[i+2] + b[i+2];
    x[j] = x[j] + b; // x[i+3] = a * x[i+3] + b[i+3];
    x[j] = x[j] + b; // x[i+4] = a * x[i+4] + b[i+4];
    x[j] = x[j] + b; // x[i+5] = a * x[i+5] + b[i+5];
    x[j] = x[j] + b; // x[i+6] = a * x[i+6] + b[i+6];
    x[j] = x[j] + b; // x[i+7] = a * x[i+7] + b[i+7];
  }
  stop_time = clock();
  *cycles = stop_time - start_time;
  //printf("%u\n", stop_time - start_time);
}

template <int C>
__global__ void axpy_simd(long n, float a, float *x, float b, uint32_t *cycles) {
  uint32_t start_time=0, stop_time=0;
  float tmp[C];
  for(int i = 0; i < C; i++) {
    tmp[i] = 0;
  }
  start_time = clock();
  for (int i = 0; i < n; i++) {
    for(int j = 0; j < C; j++)
      tmp[j] = a * tmp[j] + b;
  }
  stop_time = clock();
  for(int i = 0; i < C; i++) {
    x[i] = tmp[i];
  }
  *cycles = stop_time - start_time;
  //printf("%u\n", stop_time - start_time);
}
const int N = 1 << 25;

int main(int argc, char* argv[]) {

  struct perf_event_attr pe;
  int fd;
  uint64_t cpu_clocks;
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

  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  float *x = nullptr, *b = nullptr, *d_x = nullptr, *d_b = nullptr;
  uint32_t *d_cycles = nullptr, cycles = 0;
  x = (float*)malloc(N * sizeof(float));
  b = (float*)malloc(N * sizeof(float));
  //init
  for(int i = 0; i < N; i++) {
    x[i] = 1.0 * i;
    b[i] = 0.5 * i;
  }

  //cuda malloc
  err = cudaMalloc((void **)&d_x, N * sizeof(float));
  if (err != cudaSuccess){
      fprintf(stderr, "Failed to allocate device vector x (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void **)&d_b, N * sizeof(float));
  if (err != cudaSuccess){
      fprintf(stderr, "Failed to allocate device vector b (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void **)&d_cycles, sizeof(uint32_t));
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector b (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  //copy data to device
  err = cudaMemcpy(d_x, x,  N * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to copy vector x to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(d_b, b,  N * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to copy vector b to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  uint64_t start_cycle, end_cycle;
  struct timespec start, end;
  uint32_t bs = 1; // the block size
  uint32_t nb = 1; // the number of blocks

  //warm up
  axpy<<<1,1>>>(N,1.2, d_x, 1.0, d_cycles);
  cudaDeviceSynchronize();

  //time
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  //cpu ref clocks
  start_cycle = rdtsc();
  //cpu core clocks
  ioctl(fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);

  axpy<<<1,1>>>(N,1.2, d_x, 1, d_cycles);
  cudaDeviceSynchronize();

  ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
  end_cycle = rdtsc();
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double used_time = get_time(&start, &end);
  uint64_t used_cycles = end_cycle - start_cycle;
  double flops = 2.0 * N;
  if(read(fd, &cpu_clocks, sizeof(cpu_clocks)) == -1) {
    cout << "read cpu clocks error" << endl;
  }

  err = cudaMemcpy(&cycles, d_cycles,  sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to copy back cycles to CPU (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
    //print out
  // cout << "axpy kernel, N:       " << N << endl;
  // cout << "Flops:                " << flops << endl;
  // cout << "GPU cycles:           " << cycles << endl;
  // cout << "flops per GPU cycle:  " << flops / cycles << endl;
  // cout << "GPU clocks per iter:  " << cycles / N << endl;
  // cout << "ref cycles:           " << used_cycles << endl;
  // cout << "flops per ref cycle:  " << 1.0 * flops / used_cycles << endl;
  // cout << "cpu cycles:           " << cpu_clocks << endl;
  // cout << "flops per cpu cycle:  " << 1.0 * flops / cpu_clocks << endl;
  // cout << "cpu cycles per iter:  " << 1.0 * cpu_clocks / N << endl;
  // cout << "Ghz " << 1.0 * used_cycles /cpu_clocks << endl;
  // cout << "Achieve Gflops:        " << flops / used_time * 1e-9  << endl;

  axpy_unroll8<<<1,1>>>(N,1.2, d_x, 1, d_cycles);
  cudaDeviceSynchronize();
  err = cudaMemcpy(&cycles, d_cycles,  sizeof(uint32_t), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess){
    fprintf(stderr, "Failed to copy back cycles to CPU (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
    //print out
  // cout << "unroll 8:       " << N << endl;
  // cout << "Flops:                " << flops << endl;
  // cout << "GPU cycles:           " << cycles << endl;
  // cout << "flops per GPU cycle:  " << flops / cycles << endl;
  // cout << "GPU clocks per iter:  " << cycles / (N / 8) << endl;
  /* 1 block, 1 thread test, latency and throughput*/
  if(!strcmp("1thread", argv[1])){
    SIMD_TEST(1,1);
    SIMD_TEST(1,2);
    SIMD_TEST(1,3);
    SIMD_TEST(1,4);
    SIMD_TEST(1,5);
    SIMD_TEST(1,6);
    SIMD_TEST(1,7);
    SIMD_TEST(1,8);
  }

  /* 1 block, N thread test, latency and throughput*/
  if(!strcmp("nthreads", argv[1])){
    //C=1, N thread
    int n = 32;
    N_THREADS_TEST(n, 700,1);
    n = 32;
    N_THREADS_TEST(n, 700,2);
    n = 32;
    N_THREADS_TEST(n, 700,4);
    n = 32;
    N_THREADS_TEST(n, 700,8);

  }


  float kernel_time;
  cudaEvent_t startEvent, stopEvent;

  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  cudaEventRecord(startEvent, 0);
  axpy<<<nb,bs>>>(N,1.2, d_x, 1, d_cycles);
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&kernel_time, startEvent, stopEvent);
  //cout << "time measured in cuda:" << kernel_time << endl;

  return 0;
}
