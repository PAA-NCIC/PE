#include "../include/pe_jacobi.hpp"
#include "../include/help.hpp"
#include <iostream>
#include <cstdlib>

using namespace std;

int main() {
  int64_t jmax = 1024, kmax = 1024;
  struct timespec start, end;
  double time_used, flops, perf;
  //for simple
  double *A, *B;
  A = (double *)malloc((jmax+1) * (kmax+1) * sizeof(double));
  B = (double *)malloc((jmax+1) * (kmax+1) * sizeof(double));

  init_2d_array(A, jmax+1, kmax+1);
  init_2d_array(B, jmax+1, kmax+1);

  //warm up
  pe_jacobi2d(A, B, jmax+1, kmax+1, 1.0);
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  //call
  pe_jacobi2d(A, B, jmax+1, kmax+1, 1.0);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  time_used = get_time(&start, &end);
  flops = jmax * kmax * 4;
  perf = flops / time_used * 1e-9;
  cout << "flops:" << flops << endl;
  cout << "time : " << time_used << endl;
  cout << "Perf : " << perf << endl;
  return 0;
}
