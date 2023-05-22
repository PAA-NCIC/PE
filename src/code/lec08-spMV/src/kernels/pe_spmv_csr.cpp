#include"../../include/pe_spmv_csr.hpp"

void pe_dspmv_csr(uint32_t *row_ptr, uint32_t *colind, double *val, uint32_t m, double *x, double *y) {
  pe_spmv_csr<double>(row_ptr, colind, val, m, x, y);
}


void pe_sspmv_csr(uint32_t *row_ptr, uint32_t *colind, float *val, uint32_t m, float *x, float *y) { 
  pe_spmv_csr<float>(row_ptr, colind, val, m, x, y);
}
