#include <stdio.h>
#include <unistd.h>
#include <omp.h>
int main(int argc, char* argv[]){
  #pragma omp parallel
  {
      #pragma omp single
      {   
        printf("t%d execute work1.\n", omp_get_thread_num());
        usleep(100);
      }
      #pragma omp single
      { 
        printf("t%d execute work2.\n", omp_get_thread_num());
        usleep(100);
      }
      #pragma omp single nowait
          printf("t%d comes to end;.\n", omp_get_thread_num());
  }
  return 0;
}

