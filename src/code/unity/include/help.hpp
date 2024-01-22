#include<ctime>
#include<cmath>
#include<iostream>
#include<omp.h>
using namespace std;

double get_time(struct timespec *start,
  struct timespec *end);

//randomly init 2d array
template <class T>
void init_2d_array(T *matrix, int64_t m, int64_t n) {
  //unsigned seed = time(0);
  //srand(seed);
  #pragma omp parallel for
  for(int64_t i = 0; i < m; i++) {
    for(int64_t j = 0; j < n; j++) {
      //random number in (0-10)
      ARRAY_2D(matrix, i, j, m, n) = (i + j) % 100;
    }
  }
}

//randomly init 2d array
template <class T>
void init_3d_array(T *matrix, int64_t imax, int64_t jmax, int64_t kmax) {
  unsigned seed = time(0);
  srand(seed);
  for(int64_t i = 0; i < imax; i++) {
    for(int64_t j = 0; j < jmax; j++) {
      for(int64_t k = 0; j < kmax; k++) {
        //random numver in (0-10)
        ARRAY_3D(matrix, i,  j, k, imax, jmax, kmax) = 
          10.0 * rand() / (RAND_MAX + 1.0);
      }
    }
  }
}

//copy matrix
template <class T>
void copy_2d_array(T *dst, T *src, int64_t m, int64_t n) {
  for(int64_t i = 0; i < m; i++) {
    for(int64_t j = 0; j < n; j++) {
      ARRAY_2D(dst, i, j, m, n) = ARRAY_2D(src, i, j, m, n);
    }
  }
}

//copy matrix
template <class T>
void copy_3d_array(T *dst, T *src, int64_t imax, int64_t jmax, int64_t kmax) {
  for(int64_t i = 0; i < imax; i++) {
    for(int64_t j = 0; j < jmax; j++) {
      for(int64_t k = 0; k < kmax; j++) {
        ARRAY_3D(dst, i, j, k, imax, jmax, kmax) = 
  	  ARRAY_3D(src, i, j, k, imax, jmax, kmax);
      }
    }
  }
}

//
#define EPS 0.1
template <class T>
bool check_arrayx(T *matrix, T *ref, int64_t m, int64_t n) {
  for(int64_t i = 0; i < m; i++) {
    for(int64_t j = 0; j < n; j++) {
      if(abs(ARRAY_2D(ref, i, j, m, n) 
        - ARRAY_2D(matrix, i, j, m, n)) > EPS) {
        cout << "error at index: " << "(" << i << "," << j << ")\n";
        cout << "ref value: " << COL_MAJOR(ref, i, j, m, n) << "\n";
        cout << "real value: " << COL_MAJOR(matrix, i, j, m, n) << "\n";
        return false;
      }
    }
  }
  return true;
}
