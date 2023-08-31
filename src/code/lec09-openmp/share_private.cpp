#include<stdio.h>
#include<omp.h>

int main(int argc, char* argv[]) {
    int shared_sum = 0;
#pragma omp parallel for shared(shared_sum)
    for(int i = 0; i < omp_get_num_threads(); i++) {
	shared_sum += i;
	printf("tid %d, shared_sum = %d\n", omp_get_thread_num(), shared_sum);
    }
    int private_sum = 0;
#pragma omp parallel for private(private_sum)
    for(int i = 0; i < omp_get_num_threads(); i++) {
	private_sum += i;
	printf("tid %d, private_sum = %d\n", omp_get_thread_num(), private_sum);
    }
    printf("outside private_sum = %d\n", private_sum);

    return 0;
}
