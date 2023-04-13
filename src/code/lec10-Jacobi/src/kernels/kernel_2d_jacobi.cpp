#include"../../include/pe_jacobi.hpp"

void pe_jacobi2d(double *y, double *x, int64_t jmax, 
		 int64_t kmax, double scale) {
  pe_jacobi2d_template<double>(x, y, jmax, kmax, scale);
}

void pe_jacobi2d_blocking64(double *y, double *x, int64_t jmax,
		int64_t kmax, double scale) {
  pe_jacobi2d_blocking_template<double, 64>(x, y, jmax, kmax, scale);
}

void pe_jacobi2d_blocking(double *y, double *x, int64_t jmax,
		int64_t kmax, double scale) {
  int64_t j, k, kb;
  unsigned int kblock = 64;
  for(kb = 1; kb < kmax-1; kb += kblock) {
    for(j = 1; j < jmax-1; j++) {
      for(k = kb; k < kb + kblock; k++) {
        ARRAY_2D(y, j, k, jmax, kmax) = 
  	scale * ( ARRAY_2D(x, j-1, k, jmax, kmax) +
  	ARRAY_2D(x, j+1, k, jmax, kmax) +
  	ARRAY_2D(x, j, k-1, jmax, kmax) +
  	ARRAY_2D(x, j, k+1, jmax, kmax) );
      }
    }
  }
}


