#include"../../include/pe_gemm.hpp"

void pe_dgemm(int64_t m, int64_t n, int64_t k, 
              double *a, int64_t lda, 
              double *b, int64_t ldb,
              double *c, int64_t ldc) {
  pe_gemm<double>(m, n, k, a, lda, b, ldb, c, ldc);
}
