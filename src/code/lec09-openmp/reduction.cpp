#include <stdio.h>
#include <omp.h>
int main(int argc, char* argv[]){
    int sum = 100;
    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < 100; i++) {
      sum += i;
    }
    printf( "sum=%d\n", sum);


    //int a[100];
    //int sum[100] = {0};
    //for(int i = 0; i < 100; i++) {
    //  a[i] = i;
    //}
#pragma omp parallel for reduction(+:a)
    //for(int i = 0; i < 100; i++) 
    //  a
    return 0;
}

