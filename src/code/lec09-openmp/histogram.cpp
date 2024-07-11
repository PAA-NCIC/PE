#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 100
int main(int argc, char* argv[]){
    int IND, ID, NT;
    int S[16][8];
    int A[N];
    for(int i = 0; i < N; i++) {
      A[i] = rand() % 8;
    }
    omp_set_num_threads(16);
    #pragma omp parallel private(ID, IND)
    {
      ID = omp_get_thread_num();
      #pragma omp for nowait
      for(int i = 0; i < N; i++) {
        IND = A[i];
        S[ID][IND] = S[ID][IND] + 1;
      }
      #pragma critical 
      {
        for(int i = 0; i < 8; i++) {
          S[0][i]  = S[0][i] + S[ID][i];
        }
      }
    }

    return 0;
}

