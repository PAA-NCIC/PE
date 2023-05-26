#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "utils.cuh"
#include <helper_string.h> 
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#define CEIL_DIV(m,n) ( (m) + (n) - 1 ) / (n)
void print_matrix(const pe_f64 *A, int m, int n){
    int i;
    printf("[");
    for (i = 0; i < m * n; i++){
        if ((i + 1) % n == 0) printf("%5.2f ", A[i]);
        else printf("%5.2f, ", A[i]);
        if ((i + 1) % n == 0){
            if (i + 1 < m * n) printf(";\n");
        }
    }
    printf("]\n");
}

void randomize_matrix(pe_f64* mat, int N){
    srand(time(NULL)); int i;
    for (i = 0; i < N; i++) {
        pe_f64 tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        //tmp = i;
        mat[i] = tmp;
    }
}

double get_sec(){
    struct timeval time;
    gettimeofday(&time, NULL); 
    return (time.tv_sec + 1e-6 * time.tv_usec);
}

bool verify_matrix(pe_f64 *mat1, pe_f64 *mat2, int n){
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < n; i++){
        diff = fabs( (double)mat1[i] - (double)mat2[i] );
        if (diff > 1e-2) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i],mat2[i],i);
            return false;
        }
    }
    return true;
}

void copy_matrix(pe_f64 *src, pe_f64 *dest, int n){
    int i;
    for (i = 0; src + i && dest + i && i < n; i++) *(dest + i) = *(src + i);
    if (i != n) printf("copy failed at %d while there are %d elements in total.\n", i, n);
}


