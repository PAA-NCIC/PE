#include"../../include/pe_jacobi.hpp"

void pe_jacobi3d(double *y, double *x, int64_t imax, int64_t jmax, 
		 int64_t kmax, double scale) {
	pe_jacobi3d_template<double>(x, y, imax, jmax, kmax, scale);
}

void pe_jacobi3d_iparallel(double *y, double *x, int64_t imax, int64_t jmax, 
  int64_t kmax, double scale) {
  int64_t i, j, k;
#pragma omp for  schedule(static)
  for(i = 1; i < imax - 1; i++) {
    for(j = 1; j < jmax - 1; j++) {
      for(k = 1; k < kmax - 1; k++) {
	ARRAY_3D(y, i, j, k, imax, jmax, kmax) = scale * (
	  ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
	  ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
	  ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
	  ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
	  ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
	  ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
      }
    }
  }
}

void pe_jacobi3d_jparallel(double *y, double *x, int64_t imax, int64_t jmax, 
  int64_t kmax, double scale) {
  int64_t i, j, k;
#pragma omp for private(i)
  for(i = 1; i < imax - 1; i++) {
#pragma omp for  schedule(static)
    for(j = 1; j < jmax - 1; j++) {
      for(k = 1; k < kmax - 1; k++) {
	ARRAY_3D(y, i, j, k, imax, jmax, kmax) = scale * (
	  ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
	  ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
	  ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
	  ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
	  ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
	  ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
      }
    }
  }
}

void pe_jacobi3d_parallel_jblocking(double *y, double *x, int64_t imax, int64_t jmax, 
  int64_t kmax, double scale) {
  pe_jacobi3d_jblocking_template<double, 64>(y, x, imax, jmax, kmax, scale);
}
