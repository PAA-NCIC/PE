#include"../../include/pe_jacobi.hpp"
#include <emmintrin.h>

void pe_jacobi3d(double *y, double *x, int64_t imax, int64_t jmax, 
		 int64_t kmax, double scale) {
	pe_jacobi3d_template<double>(x, y, imax, jmax, kmax, scale);
}

void pe_jacobi3d_iparallel(double *y, double *x, int64_t imax, int64_t jmax, 
  int64_t kmax, double scale) {
  int64_t i, j, k;
  #pragma omp for schedule(static)
  for(i = 1; i < imax - 1; i++) {
    for(j = 1; j < jmax - 1; j++) {
      for(k = 1; k < kmax - 1; k++) {
	      ARRAY_3D(y, i, j, k, imax, jmax, kmax) = 
          scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
	        ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
	        ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
	        ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
	        ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
	        ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
      }
    }
  }
}

void pe_jacobi3d_iparallel_ntstore(double *y, double *x, int64_t imax, int64_t jmax, 
  int64_t kmax, double scale) {
  int64_t i, j, k;
  char *y_addr = nullptr;
   __m128i mask_all=_mm_set_epi32(-1,-1,-1,-1);
   __m128d y_vec2;
   double y1, y2;
  #pragma omp for schedule(static)
  for(i = 1; i < imax - 1; i++) {
    for(j = 1; j < jmax - 1; j++) {
      //for convenient, k shoud always be a multiple of 2
      for(k = 1; k < kmax - 1;) {
	      y_addr = (char *)&ARRAY_3D(y, i, j, k, imax, jmax, kmax);
        y1 = 
          scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
	        ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
	        ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
	        ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
	        ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
	        ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
        k++;
        y2 =
          scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
	        ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
	        ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
	        ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
	        ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
	        ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
        k++;
        y_vec2 = _mm_set_pd(y2, y1);
        //nt store [y1 y2] into memory
        _mm_maskmoveu_si128(reinterpret_cast<__m128i>(y_vec2), mask_all, y_addr);
      }
    }
  }
}


void pe_jacobi3d_jparallel(double *y, double *x, int64_t imax, int64_t jmax, 
  int64_t kmax, double scale) {
  int64_t i, j, k;
  for(i = 1; i < imax - 1; i++) {
    #pragma omp for schedule(static)
    for(j = 1; j < jmax - 1; j++) {
      for(k = 1; k < kmax - 1; k++) {
	      ARRAY_3D(y, i, j, k, imax, jmax, kmax) = 
          scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
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
