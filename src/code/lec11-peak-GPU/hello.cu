#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>

//Kernel definition
__global__ void hello() {
  printf("hello world\n");
}

int main() {
  hello<<<1,1>>>();
  return 0;
}
