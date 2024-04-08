#include<ctime>
#include<cmath>
#include<vector>
#include<iostream>
using namespace std;

double get_time(struct timespec *start,
  struct timespec *end);

//randomly generate sparse matrix
template <class T>
void init_2d_array(T *matrix, int64_t m, int64_t n) {
  unsigned seed = time(0);
  srand(seed);
  for(int64_t i = 0; i < m; i++) {
    for(int64_t j = 0; j < n; j++) {
      //random numver in (0-10)
      ARRAY_2D(matrix, i, m, j, n) = 10.0 * rand() / (RAND_MAX + 1.0);
    }
  }
}

//randomly init 2d array
template <typename T>
void init_vector(T *vec, int64_t len) {
  unsigned seed = time(0);
  srand(seed);
  for(int64_t i = 0; i < len; i++) {
	  vec[i] = 10.0 * rand() / (RAND_MAX + 1.0);
  }
}

#define EPS 0.1
template <class T>
bool check_arrayx(T *matrix, T *ref, int64_t m, int64_t n) {
  for(int64_t i = 0; i < m; i++) {
    for(int64_t j = 0; j < n; j++) {
      if(abs(ARRAY_2D(ref, i, m, j, n) 
        - ARRAY_2D(ref, i, m, j, n)) > EPS) {
        cout << "error at index: " << "(" << i << "," << j << ")\n";
        cout << "ref value: " << COL_MAJOR(ref, i, m, j, n) << "\n";
        cout << "real value: " << COL_MAJOR(matrix, i, m, j, n) << "\n";
        return false;
      }
    }
  }
  return true;
}
