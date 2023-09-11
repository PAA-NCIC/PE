#include <omp.h>
#include <iostream>
#include <cstdlib>
using namespace std;

#define N 1000
float get_lcgrand() {
  static int seed = 1;
  seed = (106 * seed + 1283) % 6075;
  return 1.0 * seed / 6075;
}

float get_lcgrand_critical(){
  static int seed = 1;
  seed = (106 * seed + 1283) % 6075;
  return 1.0 * seed / 6075;
}

float get_lcgrand_private(){
  static int seed = 1;
  seed = (106 * seed + 1283) % 6075;
  return 1.0 * seed / 6075;
}

int main(int argc, char* argv[]){
  float A[N];

  float x;
  #pragma omp parallel for private(x)
  for(int i = 0; i < N; i++) {
    //race happens
    A[i] = get_lcgrand();
  }

  float sum = 0.0;
  #pragma omp parallel for reduction(+:sum)
  for(int i = 0; i < N; i++) {
    sum = sum + A[i];
  }

  cout << "race sum: " << sum << endl;

  #pragma omp parallel for private(x)
  for(int i = 0; i < N; i++) {
    //race happens
    #pragma omp critical
    {
      A[i] = get_lcgrand_critical();
    }
  }
  sum = 0.0;
  #pragma omp parallel for reduction(+:sum)
  for(int i = 0; i < N; i++) {
    sum = sum + A[i];
  }
  cout << "no race sum: " << sum << endl;

  return 0;
}

