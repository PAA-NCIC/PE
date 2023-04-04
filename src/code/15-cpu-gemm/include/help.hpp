#include<ctime>
#include<cmath>
#include<iostream>
using namespace std;

double get_time(struct timespec *start,
  struct timespec *end);

//randomly init matrix
template <class T>
void init_matrix(T *matrix, int64_t m, int64_t n) {
  unsigned seed = time(0);
  srand(seed);
  for(int64_t i = 0; i < m; i++) {
    for(int64_t j = 0; j < n; j++) {
      //random numver in (0-10)
      COL_MAJOR(matrix, i, m, j, n) = 10.0 * rand() / (RAND_MAX + 1.0);
    }
  }
}

//copy matrix
template <class T>
void copy_matrix(T *dst, T *src, int64_t m, int64_t n) {
  for(int64_t i = 0; i < m; i++) {
    for(int64_t j = 0; j < n; j++) {
      COL_MAJOR(dst, i, m, j, n) = COL_MAJOR(src, i, m, j, n);
    }
  }
}

//
#define EPS 0.1
template <class T>
bool check_matrix(T *matrix, T *ref, int64_t m, int64_t n) {
  for(int64_t i = 0; i < m; i++) {
    for(int64_t j = 0; j < n; j++) {
      if(abs(COL_MAJOR(ref, i, m, j, n) 
        - COL_MAJOR(ref, i, m, j, n)) > EPS) {
        cout << "error at index: " << "(" << i << "," << j << ")\n";
        cout << "ref value: " << COL_MAJOR(ref, i, m, j, n) << "\n";
        cout << "real value: " << COL_MAJOR(matrix, i, m, j, n) << "\n";
        return false;
      }
    }
  }
  return true;
}
