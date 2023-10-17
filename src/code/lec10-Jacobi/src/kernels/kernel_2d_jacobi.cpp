#include"../../include/pe_jacobi.hpp"
#include<iostream>

void pe_jacobi2d(double *y, double *x, int64_t jmax, 
		 int64_t kmax, double scale, int64_t repeat) {
  pe_jacobi2d_template<double>(x, y, jmax, kmax, scale, repeat);
}

void pe_jacobi2d_blocking64(double *y, double *x, int64_t jmax,
		int64_t kmax, double scale) {
  pe_jacobi2d_blocking_template<double, 64>(x, y, jmax, kmax, scale);
}

void pe_jacobi2d_blocking(double *y, double *x, uint64_t jmax,
		uint64_t kmax, double scale, unsigned int kblock) {
  uint64_t j, k, kb;
  //update y[1,jmax][1,kmax]
  //in the later for(1<=k<=kmax), for(1<=j<=jmax)
  kmax = kmax - 2;
  jmax = jmax - 2;
  uint64_t kb_max = kmax / kblock * kblock;
  //
  //std::cout << "kmax: " << kmax << "\t" << "kblock: " << kblock << std::endl; 
  #pragma omp parallel for private(j, k) schedule(static) 
  for(kb = 1; kb < kb_max ; kb = kb + kblock) {
    for(j = 1; j <= jmax; j++) {
      for(k = kb; k < kb + kblock; k++) {
        ARRAY_2D(y, j, k, jmax, kmax) = 
        scale * ( ARRAY_2D(x, j-1, k, jmax, kmax) +
                  ARRAY_2D(x, j+1, k, jmax, kmax) +
                  ARRAY_2D(x, j, k-1, jmax, kmax) +
                  ARRAY_2D(x, j, k+1, jmax, kmax) );
      }
    }
  }
  //deal with remainder
  #pragma omp parallel for private(k) schedule(static) 
  for(j = 1; j <= jmax; j++) {
    for(k = kb_max + 1; k <= kmax; k++) {
        ARRAY_2D(y, j, k, jmax, kmax) = 
        scale * ( ARRAY_2D(x, j-1, k, jmax, kmax) +
                  ARRAY_2D(x, j+1, k, jmax, kmax) +
                  ARRAY_2D(x, j, k-1, jmax, kmax) +
                  ARRAY_2D(x, j, k+1, jmax, kmax) );
    }
  }
}

void pe_jacobi2d_blocking_parallel(double *y, double *x, uint64_t jmax,
		uint64_t kmax, double scale, unsigned int kblock) {
  uint64_t j, k, kb;
  //update y[1,jmax][1,kmax]
  //in the later for(1<=k<=kmax), for(1<=j<=jmax)
  kmax = kmax - 2;
  jmax = jmax - 2;
  uint64_t kb_max = kmax / kblock * kblock;
  //
  //std::cout << "kmax: " << kmax << "\t" << "kblock: " << kblock << std::endl; 
  for(kb = 1; kb < kb_max ; kb = kb + kblock) {
    for(j = 1; j <= jmax; j++) {
      for(k = kb; k < kb + kblock; k++) {
        ARRAY_2D(y, j, k, jmax, kmax) = 
        scale * ( ARRAY_2D(x, j-1, k, jmax, kmax) +
                  ARRAY_2D(x, j+1, k, jmax, kmax) +
                  ARRAY_2D(x, j, k-1, jmax, kmax) +
                  ARRAY_2D(x, j, k+1, jmax, kmax) );
      }
    }
  }
  //deal with remainder
  for(j = 1; j <= jmax; j++) {
    for(k = kb_max + 1; k <= kmax; k++) {
        ARRAY_2D(y, j, k, jmax, kmax) = 
        scale * ( ARRAY_2D(x, j-1, k, jmax, kmax) +
                  ARRAY_2D(x, j+1, k, jmax, kmax) +
                  ARRAY_2D(x, j, k-1, jmax, kmax) +
                  ARRAY_2D(x, j, k+1, jmax, kmax) );
    }
  }
  
}


