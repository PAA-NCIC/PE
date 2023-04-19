#ifndef _PE_DGEMM_HPP_
#define _PE_DGEMM_HPP_
#include<cstdint>
#include"macro.hpp"

template <class T>
void pe_jacobi2d_template(T *y, T *x, int64_t jmax, int64_t kmax,
		T scale){
  int64_t j, k;
    for(j = 1; j < jmax; j++) {
      for(k = 1; k < kmax; k++) {
        ARRAY_2D(y, j, jmax, k, kmax) = 
          scale * ( ARRAY_2D(x, j - 1, jmax, k, kmax) +
          ARRAY_2D(x, j + 1, jmax, k, kmax) +
          ARRAY_2D(x, j, jmax, k - 1, kmax) +
          ARRAY_2D(x, j, jmax, k + 1, kmax) );
      }
    }
}

void pe_jacobi2d(double *y, double *x, int64_t jmax, int64_t kmax,
		double scale);


#endif
