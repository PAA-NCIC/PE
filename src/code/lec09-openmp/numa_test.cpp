#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000000
int main(int argc, char* argv[]){
    double A[N] = {0}, B[N] = {0};
#pragma omp for
    for(int i = 0; i < N; i++) {
      A[i] = random() % 100;
      B[i] = A[i] * A[i];
    }
    return 0;
}

