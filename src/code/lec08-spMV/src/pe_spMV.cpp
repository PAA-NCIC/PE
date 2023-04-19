#include "../include/pe_spmv.hpp"
#include "../include/help.hpp"
#include <cstdlib>
#include <cstring>


int main() {
  struct timespec start, end;
  double time_used, flops, perf;

  int m, n, nnzA;
  int *csrColIdxA, *csrRowPtrA;
  double *csrValA, *x, *y;
  m = 6;
  n = 6;
  nnzA = 15;
  //init csr sparse matrix for simple
  csrColIdxA = (int *)malloc(nnzA * sizeof(int));
  csrValA = (double *)malloc(nnzA * sizeof(double));
  csrRowPtrA = (int *)malloc((m + 1) * sizeof(int));
  x = (double *)malloc(m * sizeof(double));
  y = (double *)malloc(m * sizeof(double));

  int row_ptr[7]     = {0,       3,                9,    11, 11, 12,      15};
  int col_idx[15]    = {0, 2, 5, 0, 1, 2, 3, 4, 5, 2, 4,      4,  2, 3, 4};
  double val[15] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  memcpy(csrRowPtrA, row_ptr, (m+1) * sizeof(int));
  memcpy(csrColIdxA, col_idx, nnzA * sizeof(int));
  memcpy(csrValA, val, nnzA * sizeof(double));
  init_vector(x, m);
  init_vector(y, m);
  //warm up
  pe_spmv_csr<double>(csrRowPtrA, csrColIdxA, csrValA, m, x, y);
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  //call
  pe_spmv_csr<double>(csrRowPtrA, csrColIdxA, csrValA, m, x, y);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  flops = 0;
  time_used = get_time(&start, &end);
  perf = flops / time_used * 1e-9;
  cout << "flops: " << flops << endl;
  cout << "time : " << time_used << endl;
  cout << "perf : " << flops / time_used << endl;
  return 0;
}
