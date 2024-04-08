#include"../../include/csr_kernel.h"

//param1: asymmetric sparse matrix in csr format
//param2: input vector
//param3: output vector
void pe_spmv_csr(CSR A, double *x, double *y) {
  int i, j;
  for (i =0; i < A.row_ptr.size()-1; i++) {
    double sum =0.0;
    for (j = A.row_ptr[i]; j < A.row_ptr[i+1]; j++) {
      sum += A.val[j] * x[A.col_idx[j]];
    }
    y[i] = sum;
  }
  return;
}

