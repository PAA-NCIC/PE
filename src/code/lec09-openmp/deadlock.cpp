#include <omp.h>
#include <iostream>
#include <cstdlib>
using namespace std;
#define N 100


int main(int argc, char* argv[]){
  omp_lock_t lock_x, lock_y;
  omp_init_lock(&lock_x);
  omp_init_lock(&lock_y);

  int x = 0, y = 0;
#pragma omp parallel 
{
#pragma omp parallel for 
  for(int i = 0; i <= N; i++) {
    if(i < 0.3 * N) {
      omp_set_lock(&lock_x);
      x = x + i;
      omp_set_lock(&lock_y);
      y = y + i;
      omp_unset_lock(&lock_y);
      omp_unset_lock(&lock_x);
    } else {
      omp_set_lock(&lock_y);
      y = y + 2 * i;
      omp_set_lock(&lock_x);
      x = x + 2 * i;
      omp_unset_lock(&lock_x);
      omp_unset_lock(&lock_y);
    }
  }
}
  
  return 0;
}
