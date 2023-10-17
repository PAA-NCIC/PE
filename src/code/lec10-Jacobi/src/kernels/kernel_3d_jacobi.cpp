#include"../../include/pe_jacobi.hpp"
#include <emmintrin.h>
#include <immintrin.h>
#include <iostream>

void pe_jacobi3d(double *y, double *x, int64_t imax, int64_t jmax, 
		 int64_t kmax, double scale) {
	pe_jacobi3d_template<double>(x, y, imax, jmax, kmax, scale);
}

void pe_jacobi3d_iparallel(double *y, double *x, int64_t imax, int64_t jmax, 
  int64_t kmax, double scale) {
  #pragma omp parallel for schedule(static)
  for(int64_t i = 1; i < imax - 1; i++) {
    for(int64_t j = 1; j < jmax - 1; j++) {
      for(int64_t k = 1; k < kmax - 1; k++) {
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

void pe_jacobi3d_iparallel_block(double *y, double *x, int64_t imax, int64_t jmax, 
  int64_t kmax, double scale, uint32_t jblock) {
  uint32_t j_multiple = (jmax - 2) / jblock * jblock;
  for(uint32_t jb=1; jb < j_multiple; jb += jblock) 
  #pragma omp parallel for schedule(static)
  for(uint32_t i = 1; i < imax - 1; i++){
    for(uint32_t j = jb; j < jb+jblock; j++){
      for(uint32_t k=1; k < kmax - 1; k++){
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
  //remainder
  #pragma omp parallel for schedule(static)
  for(uint32_t i=1; i <= imax; i++){
    for(uint32_t j = j_multiple + 1; j < jmax - 1; j++) {
      for(uint32_t k=1; k < kmax - 1; k++){
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
  #pragma omp parallel for schedule(static)
  for(i = 1; i < imax - 1; i++) {
    for(j = 1; j < jmax - 1; j++) {
      //for convenient, k shoud always be a multiple of 2
      for(k = 1; k < kmax - 1; ) {
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

void pe_jacobi3d_iparallel_block_ntstore(double *y, double *x, int64_t imax, int64_t jmax, int64_t kmax, double scale, uint32_t jblock) {
  double *y_addr = nullptr;
  __m128d y_vec2;
  double y1, y2;
  uint32_t j_multiple = (jmax - 2) / jblock * jblock;
  for(uint32_t jb=1; jb < j_multiple; jb += jblock){
    #pragma omp parallel for private(y_vec2, y1, y2) schedule(static)
    for(uint32_t i = 1; i < imax - 1; i++){
      for(uint32_t j = jb; j < jb+jblock; j++){
        //for convenient, k shoud always be a multiple of 2
        for(uint32_t k = 1; k < kmax - 1;) {
          y_addr = &ARRAY_3D(y, i, j, k, imax, jmax, kmax);
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
          //nt store [y1 y2] into memory
          y_vec2 = _mm_set_pd(y2, y1);
          _mm_stream_pd(y_addr, y_vec2);
          //_mm_store_pd(y_addr, y_vec2);
        }
     }
   }
  }
  //remainder
  //#pragma omp parallel for schedule(static)
  #pragma omp parallel for private(y_vec2, y1, y2) schedule(static)
  for(uint32_t i=1; i <= imax; i++){
    for(uint32_t j = j_multiple + 1; j < jmax - 1; j++) {
      for(uint32_t k = 1; k < kmax - 1;) {
            y_addr = &ARRAY_3D(y, i, j, k, imax, jmax, kmax);
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
            //nt store [y1 y2] into memory
            y_vec2 = _mm_set_pd(y2, y1);
            _mm_stream_pd(y_addr, y_vec2);
            //_mm_store_pd(y_addr, y_vec2);
      }
    }  
  } 
}

void pe_jacobi3d_iparallel_block_ntstore256(double *y, double *x, int64_t imax, int64_t jmax, int64_t kmax, double scale, uint32_t jblock) {
  double *y_addr = nullptr;
  __m256d y_vec4;
  double y1, y2, y3, y4;
  uint32_t j_multiple = (jmax - 2) / jblock * jblock;
  for(uint32_t jb=1; jb < j_multiple; jb += jblock){
    #pragma omp parallel for private(y_vec4, y1, y2, y3, y4) schedule(static)
    for(uint32_t i = 1; i < imax - 1; i++){
      for(uint32_t j = jb; j < jb+jblock; j++){
        //for convenient, k shoud always be a multiple of 2
        for(uint32_t k = 1; k < kmax - 1;) {
          y_addr = &ARRAY_3D(y, i, j, k, imax, jmax, kmax);
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
          y3 = 
            scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
          k++;
          y4 = 
            scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
          k++;
          //nt store [y1 y2] into memory
          y_vec4 = _mm256_set_pd(y4, y3, y2, y1);
          _mm256_store_pd(y_addr, y_vec4);
          //_mm256_stream_pd(y_addr, y_vec4);
        }
      }
    }
  }
  //remainder
  //#pragma omp parallel for schedule(static)
  #pragma omp parallel for private(y_vec4, y1, y2, y3, y4) schedule(static)
  for(uint32_t i=1; i <= imax; i++){
    for(uint32_t j = j_multiple + 1; j < jmax - 1; j++) {
      for(uint32_t k = 1; k < kmax - 1;) {
        y_addr = &ARRAY_3D(y, i, j, k, imax, jmax, kmax);
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
        y3 = 
          scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
          ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
          ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
          ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
          ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
          ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
        k++;
        y4 = 
          scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
          ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
          ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
          ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
          ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
          ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
        k++;
        //nt store [y1 y2] into memory
        y_vec4 = _mm256_set_pd(y4, y3, y2, y1);
        _mm256_store_pd(y_addr, y_vec4);
        //_mm256_stream_pd(y_addr, y_vec4);
      }
    }  
  } 
}

void pe_jacobi3d_iparallel_block_ntstore512(double *y, double *x, int64_t imax, int64_t jmax, int64_t kmax, double scale, uint32_t jblock) {
  double *y_addr = nullptr;
  __m512d y_vec8;
  double y1, y2, y3, y4, y5, y6, y7, y8;
  uint32_t j_multiple = (jmax - 2) / jblock * jblock;
  for(uint32_t jb=1; jb < j_multiple; jb += jblock){
    #pragma omp parallel for private(y_vec8, y1, y2, y3, y4, y5, y6, y7, y8) schedule(static)
    for(uint32_t i = 1; i < imax - 1; i++){
      for(uint32_t j = jb; j < jb+jblock; j++){
        //for convenient, k shoud always be a multiple of 2
        for(uint32_t k = 1; k < kmax - 1;) {
          //uint64_t offset = 0;
          // std::cout << y << std::endl;
          // std::cout << imax << " " << jmax << " " << kmax << std::endl;
          // std::cout << i << " " << j << " " << k << std::endl;
          // y_addr = (double *)((uint64_t)y + k * 8);
          // offset = k * 8;
          // std::cout << y_addr << " " << offset << std::endl;
          // offset = (k + j * kmax) * 8;
          // y_addr = (double *)((uint64_t)y + (k + j * kmax) * 8);
          // std::cout << y_addr << " " << offset << std::endl;
          // offset = (i * jmax * kmax + j * kmax + k) * 8;
          // y_addr = (double *)((uint64_t)y + (i * jmax * kmax + j * kmax + k) * 8);
          //std::cout << y_addr << " " << offset << std::endl;
          y_addr = &ARRAY_3D(y, i, j, k, imax, jmax, kmax);
          //std::cout << y_addr << std::endl;
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
          y3 = 
            scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
          k++;
          y4 = 
            scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
          k++;
          y5 =
            scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
          k++;
          y6 = 
            scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
          k++;
          y7 = 
            scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
          k++;
          y8 = 
            scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
          k++;
          //nt store [y1 y2] into memory
          y_vec8 = _mm512_set_pd(y8, y7, y6, y5, y4, y3, y2, y1);
          //_mm512_store_pd(y_addr, y_vec8);
          _mm512_stream_pd(y_addr, y_vec8);
        }
     }
   }
  }
  //remainder
  //#pragma omp parallel for schedule(static)
  #pragma omp parallel for private(y_vec8, y1, y2, y3, y4, y5, y6, y7, y8) schedule(static)
  for(uint32_t i=1; i <= imax; i++){
    for(uint32_t j = j_multiple + 1; j <= jmax; j++) {
      for(uint32_t k = 1; k < kmax - 1;) {
             y_addr = &ARRAY_3D(y, i, j, k, imax, jmax, kmax);
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
          y3 = 
            scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
          k++;
          y4 = 
            scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
          k++;
          y5 =
            scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
          k++;
          y6 = 
            scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
          k++;
          y7 = 
            scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
          k++;
          y8 = 
            scale * (ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
          k++;
          //nt store [y1 y2] into memory
          y_vec8 = _mm512_set_pd(y8, y7, y6, y5, y4, y3, y2, y1);
          //_mm512_store_pd(y_addr, y_vec8);
          _mm512_stream_pd(y_addr, y_vec8);
      }
    }  
  } 
}

void pe_jacobi3d_jparallel(double *y, double *x, int64_t imax, int64_t jmax, 
  int64_t kmax, double scale) {
  int64_t i, j, k;
  for(i = 1; i < imax - 1; i++) {
    #pragma omp parallel for schedule(static)
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
