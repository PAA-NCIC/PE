#include<stdio.h>
#include<stdlib.h>
#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
// naive version
template <typename T>
__global__  
void pe_gemm_v0(int M, int N, int K, T alpha, T* A, T* B, T beta, T* C){
    int lda = M, ldb = K, ldc = M;
    int tid_x = threadIdx.x, tid_y = threadIdx.y;
    int bid_x = blockIdx.x, bid_y = blockIdx.y;
    A = &A((bid_x<<5),0);
    B = &B(0,(bid_y<<5));
    C = &C((bid_x<<5),(bid_y<<5));
    T tmp=0.;
    for (int kk = 0; kk < K; kk++){
        tmp += A(tid_x, kk) * B(kk, tid_y);
    }
    C(tid_x,tid_y) = alpha * tmp + beta*C(tid_x,tid_y);
}
