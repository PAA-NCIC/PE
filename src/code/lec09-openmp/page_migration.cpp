#include <stdio.h>
#include <omp.h>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>
using namespace std;

const int64_t N = 30000;
const int64_t M = 30000;

double get_time(struct timespec *start,
  struct timespec *end)
{
  return end->tv_sec - start->tv_sec +
    (end->tv_nsec - start->tv_nsec) * 1e-9;
}

void dmvm(uint64_t n, uint64_t m, double *lhs, double *rhs, double *mat) {
  uint64_t offset, r, c;
  #pragma omp parallel for private(offset,c) schedule(static) 
  for(r = 0; r < n; ++r) {
    offset = m * r;
    for(c = 0; c < m; ++c) {
      lhs[r] += mat[c + offset]* rhs[c]; 
    }
  }
}


int main(int argc, char* argv[]){
  double *mat = nullptr, *x = nullptr, *y = nullptr;

  //malloc data
  mat = (double *)malloc(M * N * sizeof(double));
  x = (double *)malloc(N * sizeof(double));
  y = (double *)malloc(N * sizeof(double));

  //serial init, data should on one numa memory
  for(uint64_t i = 0; i < M * N; i++) {
    mat[i] = i % 100;
  }
  for(uint64_t i = 0; i < N; i++) {
    x[i] = y[i] = i % 10;
  }

  double time_used, flops;
  struct timespec start, end;
  int iters[] = {200, 500, 1000, 2000};
  cout << setw(10) << "iter" << "\t" << setw(10) << "threads" << "\t" << setw(10) << "perf" << endl;
  for(int nthreads = 2; nthreads <=32; nthreads = nthreads * 2) {
    for(int i = 0; i < 4; i++) {
      for(int j = 0; j < iters[i]; j++) {
        omp_set_num_threads(nthreads);
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        dmvm(N, M, y, x, mat);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        time_used = get_time(&start, &end);
        flops = M * N * 2 / time_used;
        cout << setw(10) << iters[i] << "\t" << setw(10) << nthreads << "\t" << setw(10) << setprecision(4) << flops << endl;
      }
    }
  }

  free(mat);
  free(x);
  free(y);
  return 0;
}

