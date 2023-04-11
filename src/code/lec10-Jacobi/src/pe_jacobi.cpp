#include "../include/pe_jacobi.hpp"
#include "../include/help.hpp"
#include <cstdlib>

int main() {
  int64_t m = 1024, n = 1024;
  struct timespec start, end;
  double time_used, perf;
  //for simple
  double *A, *B;
  A = (double *)malloc(m * n * sizeof(double));
  B = (double *)malloc(m * n * sizeof(double));

  init_2d_array(A, m, n);
  init_2d_array(B, m, n);

  //warm up
  pe_jacobi2d(A, B, m, n, 1.0);
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  //call
  pe_jacobi2d(A, B, m, n, 1.0);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  return 0;
}
