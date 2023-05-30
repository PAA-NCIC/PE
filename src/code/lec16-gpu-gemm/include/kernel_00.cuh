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
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    T tmp=0.;
    for (int kk = 0; kk < K; kk++){
        tmp += A(tx, kk) * B(kk, ty);
    }
    C(tx,ty) = alpha * tmp + beta*C(tx,ty);
}
