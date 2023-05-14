#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>

//Kernel definition
__global__ void hello(int date) {
  printf("hello world %d %d\n", threadIdx.x, date);
}

__global__ void bcast(int arg) {
 int value;
 if (laneId == 0) // Note unused variable for
   value = arg; // all threads except lane 0
 value = __shfl(value, 0); // Get “value” from lane 0
 if (value != arg)
   printf(“Thread %d failed.\n”, threadIdx.x);
}

int main() {
  hello<<<1,1>>>(2023);
  bcast<<<1,32>>>(1234)
  return 0;
}
