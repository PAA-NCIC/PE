#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>
using namespace std;

#define phi(A, i,j,k) A[(i) * jmax * kmax + (j) * kmax + (k)]


void  GaussSeidel(double *A, double osth , uint64_t iter, uint64_t imax, uint64_t jmax, uint64_t kmax) {
  for(uint64_t it = 0; it < iter; it++) {
    for(uint64_t k = 1; k < kmax-1; k++) {
      for(uint64_t j = 1; j < jmax-1; j++) {
        for(uint64_t i = 1; i < imax-1; i++) {
          phi(A, i, j, k) = ( phi(A, i-1, j,   k) +   phi(A, i+1, j,   k)
                            + phi(A, i,   j-1, k) +   phi(A, i,   j+1, k)
                            + phi(A, i,   j,   k-1) + phi(A, i,   j,   k+1) )* osth;
        }
      }
    }
  }
}


void  GaussSeidelParallel(double *A,   double osth , uint64_t iter, uint64_t imax, uint64_t jmax, uint64_t kmax) {
  int tid, numthreads;
  uint64_t it, i, j, k, jStart, jEnd;
  for(it = 0; it < iter; it++) {
    #pragma omp parallel private(tid, i, j, k, jStart, jEnd) 
    {
      tid = omp_get_thread_num();
      #pragma omp single
      {
        numthreads = omp_get_num_threads();
        cout << "numthreads : " << numthreads << endl;
      }//default barrier for all threads
      jStart = jmax / numthreads * tid + 1;
      jEnd = jStart + jmax / numthreads;
      for(uint64_t l = 1; l < kmax + numthreads - 1; l++) {
        k = l - tid;
        if(1 <= k < kmax - 1) {
          for(j = jStart; j <= jEnd; j++) { 
            for(i = 1; i < imax - 1; i++) {
              phi(A, i, j, k) = ( phi(A, i-1, j,  k)    + phi(A, i+1, j,   k)
                                + phi(A, i,   j-1, k)   + phi(A, i,   j+1, k)
                                + phi(A, i,   j,   k-1) + phi(A, i,   j,   k+1) ) * osth;
            }
          }
        }
      }
    }
    
  }
}

double get_time(struct timespec *start,
  struct timespec *end)
{
  return end->tv_sec - start->tv_sec +
    (end->tv_nsec - start->tv_nsec) * 1e-9;
}

#define N (imax * jmax * kmax)
int main(int argc, char* argv[]){
  uint64_t imax = 1024, jmax = 1681, kmax = 1024;
  //uint64_t imax = 16, jmax = 17, kmax = 16;

  double *A = nullptr;
  A = (double *)malloc(N * sizeof(double));

  double time_used, lups, perf;
  struct timespec start, end;

  #pragma omp for
  for(int i = 0; i < N; i++) {
    A[i] = random() % 100;
  }

  cout << "imax:" << imax << ", jmax:" << jmax << ", kmax:" << kmax << endl;
  cout << setw(12) << "threadsnum" << "\t" << setw(10) <<"lup" << endl;

  //serial GaussSeidel
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  GaussSeidel(A, 1/6.0, 1, imax, jmax, kmax);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  time_used = get_time(&start, &end);
  lups = (imax - 2) * (jmax - 2) * (kmax - 2);
  perf = 1.0 * lups / time_used * 1e-6;  // unit MLUP/s
  cout << setw(12) << "1" << "\t" << setprecision(4) << perf << endl;

  //parallel GaussSeidel
  for(int threadnum = 2; threadnum <= 16; threadnum +=2 ) {
    omp_set_num_threads(threadnum);
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    GaussSeidelParallel(A, 1/6.0, 1, imax, jmax, kmax);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    time_used = get_time(&start, &end);
    lups = (imax - 2) * (jmax - 2) * (kmax - 2);
    perf = 1.0 * lups / time_used * 1e-6;  // unit MLUP/s
    cout << setw(12) << threadnum << "\t" << setprecision(4) << perf << endl;
  }
  return 0;
}

