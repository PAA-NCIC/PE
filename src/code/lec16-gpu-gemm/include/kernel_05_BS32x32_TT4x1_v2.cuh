#include<stdio.h>
#include<stdlib.h>

//shared memory blocking
//transform B in shared memory to optimize shared memory visit
template <typename T,
          unsigned int MS,
          unsigned int NS,
          unsigned int KS>
__global__  
void pe_gemm_v5(int M, int N, int K, T alpha, T* A, T* B, T beta, T* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int row0 = (tx & 0x7) << 1 , col = tx >> 3;
    int row1 = row0 + 16;
    int bx = blockIdx.x, by = blockIdx.y;
    T* A_bptr = &A[bx * MS];
    T* B_bptr = &B[by * NS * ldc];
    T* C_bptr = &C[bx * MS + by * NS * ldc];
    __shared__ T sA[MS*KS];
    __shared__ T sB[KS*NS];
    T C_reg[4] = {0.0, 0.0, 0.0, 0.0};
    for (int kb = 0; kb < K; kb += KS){
        //column-major storage
        sA[(row0 + 0) + col * MS] = A_bptr[(row0  + 0) + col * lda];
        sA[(row0 + 1) + col * MS] = A_bptr[(row0 + 1) + col * lda];
        sA[(row1 + 0) + col * MS] = A_bptr[(row1 + 0) + col * lda];
        sA[(row1 + 1) + col * MS] = A_bptr[(row1 + 1) + col * lda];
        //transform B in shared memory
        //sB[row  * KS + col] = B_bptr[row  + col * ldb];
        //sB[row1 * KS + col] = B_bptr[row1 + col * ldb];
        //sB[row2 * KS + col] = B_bptr[row2 + col * ldb];
        //sB[row3 * KS + col] = B_bptr[row3 + col * ldb];
        sB[(row0 + 0) + col * KS] = B_bptr[(row0 + 0) + col * ldb];
        sB[(row0 + 1) + col * KS] = B_bptr[(row0 + 1) + col * ldb];
        sB[(row1 + 0) + col * KS] = B_bptr[(row1 + 0) + col * ldb];
        sB[(row1 + 1) + col * KS] = B_bptr[(row1 + 1) + col * ldb];
        A_bptr += lda * KS;
        B_bptr += KS;
        __syncthreads();
        for (int kk = 0; kk < KS; kk++){
            //T b_reg = sB[col + kk * KS];
            T b_reg = sB[col * KS + kk];
            C_reg[0] += sA[(row0 + 0) + kk * MS] * b_reg;
            C_reg[1] += sA[(row0 + 1) + kk * MS] * b_reg;
            C_reg[2] += sA[(row1 + 0) + kk * MS] * b_reg;
            C_reg[3] += sA[(row1 + 1) + kk * MS] * b_reg;
        }
        __syncthreads();
    }
    C_bptr[(row0 + 0) + col * ldc] = alpha * C_reg[0] + beta * C_bptr[(row0 + 0) + col * ldc];
    C_bptr[(row0 + 1) + col * ldc] = alpha * C_reg[1] + beta * C_bptr[(row0 + 1) + col * ldc];
    C_bptr[(row1 + 0) + col * ldc] = alpha * C_reg[2] + beta * C_bptr[(row1 + 0) + col * ldc];
    C_bptr[(row1 + 1) + col * ldc] = alpha * C_reg[3] + beta * C_bptr[(row1 + 1) + col * ldc];
}
