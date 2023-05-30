#include<stdio.h>
#include<stdlib.h>

//shared memory blocking
//transform B in shared memory to optimize shared memory visit
template <typename T,
          unsigned int MS,
          unsigned int NS,
          unsigned int KS>
__global__  
void pe_gemm_v6(int M, int N, int K, T alpha, T* A, T* B, T beta, T* C){
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
    double2 c_reg[2];
    c_reg[0] = *(double2 *)(C_bptr + row0 + col * ldc);
    c_reg[1] = *(double2 *)(C_bptr + row1 + col * ldc);
    c_reg[0].x = beta * c_reg[0].x;
    c_reg[0].y = beta * c_reg[0].y;
    c_reg[1].x = beta * c_reg[1].x;
    c_reg[1].y = beta * c_reg[1].y;
    for (int kb = 0; kb < K; kb += KS){
        //vector load
         *(double2*)(sA + row0 + col * MS) = 
           *(double2*)(A_bptr + row0 + col * lda);
         *(double2*)(sA + row1 + col * MS) = 
           *(double2*)(A_bptr + row1 + col * lda);
        //transform B in shared memory
        //sB[row  * KS + col] = B_bptr[row  + col * ldb];
        //sB[row1 * KS + col] = B_bptr[row1 + col * ldb];
        //sB[row2 * KS + col] = B_bptr[row2 + col * ldb];
        //sB[row3 * KS + col] = B_bptr[row3 + col * ldb];
        *(double2*)(sB + row0 + col * KS) = 
          *(double2*)(B_bptr + row0 + col * ldb);
        *(double2*)(sB + row1 + col * KS) = 
          *(double2*)(B_bptr + row1 + col * ldb);
        A_bptr += lda * KS;
        B_bptr += KS;
        __syncthreads();
        for (int kk = 0; kk < KS; kk++){
            //T b_reg = sB[col + kk * KS];
            T b_reg = sB[col * KS + kk];
            c_reg[0].x += sA[(row0 + 0) + kk * MS] * b_reg;
            c_reg[0].y += sA[(row0 + 1) + kk * MS] * b_reg;
            c_reg[1].x += sA[(row1 + 0) + kk * MS] * b_reg;
            c_reg[1].y += sA[(row1 + 1) + kk * MS] * b_reg;
        }
        __syncthreads();
    }
    *(double2 *)(C_bptr + row0 + col * ldc) = c_reg[0];
    *(double2 *)(C_bptr + row1 + col * ldc) = c_reg[1];
}
