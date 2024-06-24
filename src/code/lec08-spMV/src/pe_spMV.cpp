#include "../include/csr_formatter.h"
#include "../include/csr_kernel.h"
#include "../include/help.hpp"
#include <cstdlib>
#include <cstring>


int main(int argc, char* argv[]) {
  struct timespec start, end;
  double time_used, flops, perf;

  if(argc == 1) {
    int m, n, nnzA;
    int *csrColIdxA = nullptr, *csrRowPtrA = nullptr;
    double *csrValA = nullptr, *x = nullptr, *y = nullptr;
    m = 6;
    n = 6;
    nnzA = 15;
    //init csr sparse matrix for simple
    csrColIdxA = (int *)malloc(nnzA * sizeof(uint32_t));
    csrValA = (double *)malloc(nnzA * sizeof(double));
    csrRowPtrA = (int *)malloc((m + 1) * sizeof(uint32_t));
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
    //transform into CSR
    CSR A;
    A.row_ptr = std::vector<int>(row_ptr, row_ptr + sizeof(row_ptr) / sizeof(int));
    A.col_idx = std::vector<int>(col_idx, col_idx + sizeof(col_idx) / sizeof(int));
    A.val = std::vector<double>(val, val + sizeof(val) / sizeof(double));
    //warm up
    pe_spmv_csr(A, x, y);
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    //call
    pe_spmv_csr(A, x, y);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    flops = nnzA * 2;
    time_used = get_time(&start, &end);
    perf = flops / time_used * 1e-9;
    cout << "flops: " << flops << endl;
    cout << "time : " << time_used << endl;
    cout << "perf : " << flops / time_used << endl;
  } 
  else { // read from a mtk file
    double *x = nullptr, *y=nullptr;
    if(argc == 3 && std::strcmp(argv[2], "symmetric") == 0) {
      //load symmetric matrix
      std::cout << "load symmetric matrix\n";
      CSR A = assemble_symmetric_csr_matrix(argv[1]);
      cout << "Matrix bandwidth is " << getBandwidth(A) << '\n';
      //printMatrix(A);
      //for(int i = 0; i < A.col_idx.size(); i++) {
      //  cout << A.col_idx[i] << "\n";
      //}
      x = (double *)malloc(A.cols * sizeof(double));
      y = (double *)malloc(A.cols * sizeof(double));
      init_vector(x, A.cols);
      init_vector(y, A.cols);
      flops = A.nnz * 2;
      pe_spmv_csr(A, x, y);
      clock_gettime(CLOCK_MONOTONIC_RAW, &start);
      pe_spmv_csr(A, x, y);
      clock_gettime(CLOCK_MONOTONIC_RAW, &end);
      time_used = get_time(&start, &end);
      perf = flops / time_used * 1e-9;
      cout << "flops: " << flops << endl;
      cout << "time : " << time_used << endl;
      cout << "perf : " << flops / time_used << endl;
    } else {
      std::cout << "load asymmetric matrix\n";
      CSR A = assemble_csr_matrix(argv[1]);
      cout << "Matrix bandwidth is " << getBandwidth(A) << '\n';
      x = (double *)malloc(A.cols * sizeof(double));
      y = (double *)malloc(A.cols * sizeof(double));
      init_vector(x, A.cols);
      init_vector(y, A.cols);
      flops = A.nnz * 2;
      clock_gettime(CLOCK_MONOTONIC_RAW, &start);
      pe_spmv_csr(A, x, y);
      clock_gettime(CLOCK_MONOTONIC_RAW, &end);
      time_used = get_time(&start, &end);
      perf = flops / time_used * 1e-9;
      cout << "flops: " << flops << endl;
      cout << "time : " << time_used << endl;
      cout << "perf : " << flops / time_used << endl;
    }
  }

  return 0;
}
