#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>

//Kernel definition
__global__ void hello(int date) {
  printf("hello world %d %d\n", threadIdx.x, date);
}


int main() {
  hello<<<1,1>>>(2023);
  cudaDeviceSynchronize();
  return 0;
}
