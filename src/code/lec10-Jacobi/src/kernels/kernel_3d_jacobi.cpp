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
          scale * (
          ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
	        ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
	        ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
	        ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
	        ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
	        ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
      }
    }
  }
}

void pe_jacobi3d_iparallel_block(double *y, double *x, int64_t imax, int64_t jmax, 
  int64_t kmax, double scale, uint32_t jblock) {
  uint32_t j_multiple = (jmax - 2) / jblock * jblock;
  for(uint32_t jb=1; jb < j_multiple; jb += jblock){
  #pragma omp parallel for schedule(static)
    for(uint32_t i = 1; i < imax - 1; i++){
      for(uint32_t j = jb; j < jb+jblock; j++){
        for(uint32_t k=1; k < kmax - 1; k++){
          ARRAY_3D(y, i, j, k, imax, jmax, kmax) = 
            scale * (
            ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
            ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
            ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
            ARRAY_3D(x, i+1, j, k, imax, jmax, kmax) );
        }
      }
    }	
  } 
  //remainder
  #pragma omp parallel for schedule(static)
  for(uint32_t i=1; i < imax-1; i++){
    for(uint32_t j = j_multiple + 1; j < jmax - 1; j++) {
      for(uint32_t k=1; k < kmax - 1; k++){
        ARRAY_3D(y, i, j, k, imax, jmax, kmax) = 
          scale * (
          ARRAY_3D(x, i, j, k-1, imax, jmax, kmax) +
          ARRAY_3D(x, i, j, k+1, imax, jmax, kmax) +
          ARRAY_3D(x, i, j-1, k, imax, jmax, kmax) +
          ARRAY_3D(x, i, j+1, k, imax, jmax, kmax) +
          ARRAY_3D(x, i-1, j, k, imax, jmax, kmax) +
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

void pe_jacobi3d_iparallel_block_ntstore(double *y, double *x, int64_t imax, 
  int64_t jmax, int64_t kmax, double scale, uint32_t jblock) {
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

void pe_jacobi3d_iparallel_block_ntstore256(double *y, double *x, int64_t imax,
  int64_t jmax, int64_t kmax, double scale, uint32_t jblock) {
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

void pe_jacobi3d_iparallel_block_ntstore512(double *y, double *x, int64_t imax,
  int64_t jmax, int64_t kmax, double scale, uint32_t jblock) {
  __m512d y_vec8, v_scale;
  __m512d x1,x2,x3,x4,x5,x6;
  v_scale = _mm512_broadcastsd_pd( _mm_set_sd(scale));
  uint32_t j_multiple = (jmax - 2) / jblock * jblock;
  for(uint32_t jb=1; jb < j_multiple; jb += jblock){
    #pragma omp parallel for private(y_vec8, v_scale, x1, x2, x3, x4, x5, x6) schedule(static)
    for(uint32_t i = 1; i < imax - 1; i++){
      for(uint32_t j = jb; j < jb+jblock; j++){
        double *y_addr = &ARRAY_3D(y, i, j, 1, imax, jmax, kmax);
        //std::cout << "y_addr:" << y_addr << std::endl;
        double *x_addr = &ARRAY_3D(x, i, j, 0, imax, jmax, kmax);
        //k should be a multiple of 64 bytes for aligned purpose
        //for simple demonstration, we only deal with aligned parts 
        //we leave the boundary for you 
        //or you can regard the remaining 6 elements [k-6, k-1], as padding
        for(uint32_t k = 1; k < kmax - 1; k += 8) {
          x1 = _mm512_loadu_pd(x_addr - jmax * kmax + k);   //x[i-1, j, k] 
          x2 = _mm512_loadu_pd(x_addr - kmax + k - 1);      //x[i,j-1,k-1]
          x3 = _mm512_loadu_pd(x_addr + k - 1);             //x[i,j,k-1]
          x4 = _mm512_loadu_pd(x_addr + k + 1);             //x[i,j,k+1]
          x5 = _mm512_loadu_pd(x_addr + kmax + k);          //x[i,j+1,k]
          x6 = _mm512_loadu_pd(x_addr + jmax * kmax + k);   //x[i+1,j,k]
          y_vec8 = x1 + x2 + x3 + x4 + x5 +x6;
          //std::cout << "y_addr:" << y_addr << std::endl;
          y_vec8 = _mm512_mul_pd(y_vec8, v_scale);
          //y_addr must be aligned on a 64-byte boundary 
          _mm512_stream_pd(y_addr, y_vec8);
          y_addr += 8;
        }
     }
   }
  }
  //remainder
  //#pragma omp parallel for schedule(static)
  #pragma omp parallel for private(y_vec8, v_scale, x1, x2, x3, x4, x5, x6) schedule(static)
  for(uint32_t i=1; i < imax - 1; i++){
    for(uint32_t j = j_multiple + 1; j < jmax - 1; j++) {
      double *y_addr = &ARRAY_3D(y, i, j, 1, imax, jmax, kmax);
      double *x_addr = &ARRAY_3D(x, i, j, 0, imax, jmax, kmax);
      for(uint32_t k = 1; k < kmax - 1; k+=8) {
          x1 = _mm512_loadu_pd(x_addr - jmax * kmax + k);     //x[i-1, j, k] 
          x2 = _mm512_loadu_pd(x_addr - kmax + k - 1);        //x[i,j-1,k-1]
          x3 = _mm512_loadu_pd(x_addr + k - 1);               //x[i,j,k-1]
          x4 = _mm512_loadu_pd(x_addr + k + 1);               //x[i,j,k+1]
          x5 = _mm512_loadu_pd(x_addr + kmax + k);            //x[i,j+1,k]
          x6 = _mm512_loadu_pd(x_addr + jmax * kmax + k);     //x[i+1,j,k]
          y_vec8 = x1 + x2 + x3 + x4 + x5 +x6;
          y_vec8 = _mm512_mul_pd(y_vec8, v_scale);
          //_mm512_store_pd(y_addr, y_vec8);
          _mm512_stream_pd(y_addr, y_vec8);
          y_addr += 8;
      }
    }  
  } 
}

void pe_jacobi3d_iparallel_block_store512(double *y, double *x, int64_t imax, int64_t jmax,
  int64_t kmax, double scale, uint32_t jblock) {
  __m512d y_vec8, v_scale;
  __m512d x1,x2,x3,x4,x5,x6;
  v_scale = _mm512_broadcastsd_pd( _mm_set_sd(scale));
  uint32_t j_multiple = (jmax - 2) / jblock * jblock;
  for(uint32_t jb=1; jb < j_multiple; jb += jblock){
    #pragma omp parallel for private(y_vec8, v_scale, x1, x2, x3, x4, x5, x6) schedule(static)
    for(uint32_t i = 1; i < imax - 1; i++){
      for(uint32_t j = jb; j < jb+jblock; j++){
        double *y_addr = &ARRAY_3D(y, i, j, 1, imax, jmax, kmax);
        //std::cout << "y_addr:" << y_addr << std::endl;
        double *x_addr = &ARRAY_3D(x, i, j, 0, imax, jmax, kmax);
        //for convenient, k shoud always be a multiple of 2
        for(uint32_t k = 1; k < kmax - 1; k += 8) {
          x1 = _mm512_loadu_pd(x_addr - jmax * kmax + k);//x[i-1, j, k] 
          x2 = _mm512_loadu_pd(x_addr - kmax + k - 1);//x[i,j-1,k-1]
          x3 = _mm512_loadu_pd(x_addr + k - 1);//x[i,j,k-1]
          x4 = _mm512_loadu_pd(x_addr + k + 1);//x[i,j,k+1]
          x5 = _mm512_loadu_pd(x_addr + kmax + k);//x[i,j+1,k]
          x6 = _mm512_loadu_pd(x_addr + jmax * kmax + k);//x[i+1,j,k]
          y_vec8 = x1 + x2 + x3 + x4 + x5 +x6;
          //std::cout << "y_addr:" << y_addr << std::endl;
          y_vec8 = _mm512_mul_pd(y_vec8, v_scale);
          _mm512_store_pd(y_addr, y_vec8);
          //_mm512_stream_pd(y_addr, y_vec8);
          y_addr += 8;
        }
     }
   }
  }
  //remainder
  //#pragma omp parallel for schedule(static)
  #pragma omp parallel for private(y_vec8, v_scale, x1, x2, x3, x4, x5, x6) schedule(static)
  for(uint32_t i=1; i < imax - 1; i++){
    for(uint32_t j = j_multiple + 1; j < jmax - 1; j++) {
      double *y_addr = &ARRAY_3D(y, i, j, 1, imax, jmax, kmax);
      double *x_addr = &ARRAY_3D(x, i, j, 0, imax, jmax, kmax);
      for(uint32_t k = 1; k < kmax - 1; k+=8) {
          x1 = _mm512_loadu_pd(x_addr - jmax * kmax + k);//x[i-1, j, k] 
          x2 = _mm512_loadu_pd(x_addr - kmax + k - 1);//x[i,j-1,k-1]
          x3 = _mm512_loadu_pd(x_addr + k - 1);//x[i,j,k-1]
          x4 = _mm512_loadu_pd(x_addr + k + 1);//x[i,j,k+1]
          x5 = _mm512_loadu_pd(x_addr + kmax + k);//x[i,j+1,k]
          x6 = _mm512_loadu_pd(x_addr + jmax * kmax + k);//x[i+1,j,k]
          y_vec8 = x1 + x2 + x3 + x4 + x5 +x6;
          y_vec8 = _mm512_mul_pd(y_vec8, v_scale);
          _mm512_store_pd(y_addr, y_vec8);
          //_mm512_stream_pd(y_addr, y_vec8);
          y_addr += 8;
      }
    }  
  } 
}

void pe_jacobi3d_iparallel_store512(double *y, double *x, int64_t imax, int64_t jmax, int64_t kmax,
  double scale) 
{
  __m512d y_vec8, v_scale;
  __m512d x1,x2,x3,x4,x5,x6;
  v_scale = _mm512_broadcastsd_pd( _mm_set_sd(scale));
  #pragma omp parallel for private(y_vec8, v_scale, x1, x2, x3, x4, x5, x6) schedule(static)
  for(uint32_t i = 1; i < imax - 1; i++){
    for(uint32_t j = 1; j < jmax; j++){
      double *y_addr = &ARRAY_3D(y, i, j, 1, imax, jmax, kmax);
      //std::cout << "y_addr:" << y_addr << std::endl;
      double *x_addr = &ARRAY_3D(x, i, j, 0, imax, jmax, kmax);
      //for convenient, k shoud always be a multiple of 2
      for(uint32_t k = 1; k < kmax - 1; k += 8) {
        x1 = _mm512_loadu_pd(x_addr - jmax * kmax + k);//x[i-1, j, k] 
        x2 = _mm512_loadu_pd(x_addr - kmax + k - 1);//x[i,j-1,k-1]
        x3 = _mm512_loadu_pd(x_addr + k - 1);//x[i,j,k-1]
        x4 = _mm512_loadu_pd(x_addr + k + 1);//x[i,j,k+1]
        x5 = _mm512_loadu_pd(x_addr + kmax + k);//x[i,j+1,k]
        x6 = _mm512_loadu_pd(x_addr + jmax * kmax + k);//x[i+1,j,k]
        y_vec8 = x1 + x2 + x3 + x4 + x5 +x6;
        //std::cout << "y_addr:" << y_addr << std::endl;
        y_vec8 = _mm512_mul_pd(y_vec8, v_scale);
        _mm512_store_pd(y_addr, y_vec8);
        //_mm512_stream_pd(y_addr, y_vec8);
        y_addr += 8;
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

void pe_jacobi3d_kparallel(double *y, double *x, int64_t imax, int64_t jmax, 
  int64_t kmax, double scale) {
  int64_t i, j, k;
  for(i = 1; i < imax - 1; i++) {
    for(j = 1; j < jmax - 1; j++) {
      #pragma omp parallel for schedule(static)
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

void pe_nt_bw(double *y, double *x, int64_t imax, int64_t jmax, 
  int64_t kmax, double scale) {
  int64_t size = imax * jmax * kmax & 0xFFFFFFF8;
  __m512d y_tmp = _mm512_set1_pd(scale);
  #pragma omp parallel for schedule(static)
  for(int64_t i = 0; i < size - 1; i += 32) {
    _mm512_stream_pd(y+i, y_tmp);
    _mm512_stream_pd(y+i+8, y_tmp);
    _mm512_stream_pd(y+i+16, y_tmp);
    _mm512_stream_pd(y+i+24, y_tmp);
  }
}


void pe_store_bw(double *y, double *x, int64_t imax, int64_t jmax, 
  int64_t kmax, double scale) {
  int64_t size = imax * jmax * kmax & 0xFFFFFFF8;
  __m512d y_tmp = _mm512_set1_pd(scale);
  #pragma omp parallel for schedule(static)
  for(int64_t i = 0; i < size - 1; i += 32) {
      _mm512_store_pd(y+i, y_tmp);
      _mm512_store_pd(y+i+8, y_tmp);
      _mm512_store_pd(y+i+16, y_tmp);
      _mm512_store_pd(y+i+24, y_tmp);
  }
}

void pe_jacobi3d_parallel_jblocking(double *y, double *x, int64_t imax, int64_t jmax, 
  int64_t kmax, double scale) {
  pe_jacobi3d_jblocking_template<double, 64>(y, x, imax, jmax, kmax, scale);
}
