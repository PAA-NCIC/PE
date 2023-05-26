#include <stdio.h>
#include <stdlib.h>
#include <helper_string.h> 
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <sys/time.h>
#include "utils.cuh"
#include "include/kernels.h"
#define MYSGEMM mysgemm_naive // select the kernel here
typedef void (*pe_dgemm_kernel)(int, int, int, pe_f64, pe_f64 *, pe_f64 *, pe_f64, pe_f64 *);
const int N_KERNELS = 10;
const int N_CASES = 4;
pe_dgemm_kernel PE_DGEMM_KERNELS[N_KERNELS] = {
  pe_dgemm_v0,
  pe_dgemm_v0,
  pe_dgemm_v0,
  pe_dgemm_v0,
  pe_dgemm_v0,
  pe_dgemm_v0,
  pe_dgemm_v0,
  pe_dgemm_v0,
  pe_dgemm_v0,
  pe_dgemm_v0
};

__global__ void vectorAdd(pe_f64 *a, pe_f64 *b, pe_f64 *result, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        result[tid] = a[tid] + b[tid];
    }
}

int main(int argc, char **argv){
    if (argc != 2) {
        printf("Please select a kernel (range 0 - 10).\n");
        exit(-1);
    }
    //1024-10240-256
    int SIZE[N_CASES];
    for (int i=0;i<N_CASES;i++) 
      SIZE[i]=(i+4)<<8;
    int kernel_id=atoi(argv[1]);
    if (kernel_id<0||kernel_id>11) {
        printf("Please enter a valid kernel number (0-11).\n");
        exit(-2);
    }
    int m, n, k,max_size;
    int repeat=5;
    max_size=SIZE[N_CASES-1];
    pe_f64 *A=NULL,*B=NULL,*C=NULL,*C_ref=NULL;//host matrices
    pe_f64 *dA=NULL,*dB=NULL,*dC=NULL,*dC_ref=NULL;//device matrices
    pe_f64 alpha = 1.0, beta = 0.;//two arbitary input parameters
    float elapsed_time;
    cublasHandle_t err; cublasCreate(&err);
    //malloc memory on the host
    A=(pe_f64 *)malloc(sizeof(pe_f64)*max_size*max_size);
    B=(pe_f64 *)malloc(sizeof(pe_f64)*max_size*max_size);
    C=(pe_f64 *)malloc(sizeof(pe_f64)*max_size*max_size);
    C_ref=(pe_f64 *)malloc(sizeof(pe_f64)*max_size*max_size);
    //init matrices
    randomize_matrix(A,max_size*max_size);
    randomize_matrix(B,max_size*max_size);
    randomize_matrix(C,max_size*max_size);
    copy_matrix(C,C_ref,max_size*max_size);
    //malloc memory on the GPU
    CUDA_CALLER(cudaMalloc((void**) &dA, sizeof(pe_f64)*max_size*max_size));
    CUDA_CALLER(cudaMalloc((void**) &dB, sizeof(pe_f64)*max_size*max_size));
    CUDA_CALLER(cudaMalloc((void**) &dC, sizeof(pe_f64)*max_size*max_size));
    CUDA_CALLER(cudaMalloc((void**) &dC_ref, sizeof(pe_f64)*max_size*max_size));
    //host to device
    CUDA_CALLER(cudaMemcpy(dA, A, sizeof(pe_f64)*max_size*max_size, cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dB, B, sizeof(pe_f64)*max_size*max_size, cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dC, C, sizeof(pe_f64)*max_size*max_size, cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dC_ref, C_ref, sizeof(pe_f64)*max_size*max_size, cudaMemcpyHostToDevice));

    cudaEvent_t eventStart, eventEnd;
    cudaEventCreate(&eventStart);
    cudaEventCreate(&eventEnd);
    for (int size_item=0; size_item<N_CASES; size_item++){
      m=n=k=SIZE[size_item];
      printf("\nM=N=K=%d:\n",m);
      //correctness verification
      {
          cublasDgemm(err, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                     const_cast<const double*>(&alpha), \
                     const_cast<const double*>(dA), m, \
                     const_cast<const double*>(dB), k, \
                     const_cast<const double*>(&beta), dC_ref, m);
          CUDA_KERNEL_CALLER(PE_DGEMM_KERNELS[kernel_id](m,n,k,alpha,dA,dB,beta,dC));
          cudaDeviceSynchronize();
          cudaMemcpy(C, dC, sizeof(pe_f64)*m*n, cudaMemcpyDeviceToHost);
          cudaMemcpy(C_ref, dC_ref, sizeof(pe_f64)*m*n, cudaMemcpyDeviceToHost);
          cudaDeviceSynchronize();
          if (!verify_matrix(C_ref,C,m*n)) {
              printf("Failed to pass the correctness verification against NVIDIA cuBLAS. Exited.\n");
              exit(-3);
          }
      }
      //bench performance
      {
        cudaEventRecord(eventStart);
        for (int i=0;i < repeat; i++){
          PE_DGEMM_KERNELS[kernel_id](m,n,k,alpha,dA,dB,beta,dC);
          //cublasDgemm(err, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
          //           const_cast<const double*>(&alpha), \
          //           const_cast<const double*>(dA), m, \
          //           const_cast<const double*>(dB), k, \
          //           const_cast<const double*>(&beta), dC_ref, m);
         //CUDA_KERNEL_CALLER(vectorAdd<<<32, 32>>>(dA, dB, dC, m * n));
        }
        cudaEventRecord(eventEnd);
        //cudaEventSynchronize(eventStart);
        cudaEventSynchronize(eventEnd);
        cudaEventElapsedTime(&elapsed_time, eventStart, eventEnd);
        elapsed_time /= 1000.;

        printf("Average elasped time: %f second, performance: %f GFLOPS.\n", elapsed_time/repeat,2.*1e-9*m*n*k/elapsed_time);
        fflush(stdout);
        copy_matrix(C_ref,C,m*n);//sync C with cuBLAS to prepare for the next run
      }
    }
    cudaDeviceSynchronize();
    free(A);free(B);free(C);free(C_ref);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dC_ref);
    cudaDeviceSynchronize();
    return 0;
}
