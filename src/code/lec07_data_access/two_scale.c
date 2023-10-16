#include "util.h"

void data_init(double** A_p, double** B_p, double** C_p, uint64_t len){
    *A_p = (double*)aligned_alloc(64, len * sizeof(double));
    *B_p = (double*)aligned_alloc(64, len * sizeof(double));
    *C_p = (double*)aligned_alloc(64, len * sizeof(double));
    
    double* A = *A_p;
    double* B = *B_p;
    double* C = *C_p;

    for(uint64_t i = 0; i < len; ++i){
        A[i] = 0.;
        B[i] = 1.;
        C[i] = 1.;
    }
}

void data_release(double* A, double* B, double* C){
    free(A);
    free(B);
    free(C);
}

void two_scale_kernel_naive(double* A, double p,double *B, double q, double* C, uint64_t index_region){
    for(uint64_t i = 0; i < index_region; ++i){
        A[i] = p * B[i];
    }
    for(uint64_t i = 0; i < index_region; ++i){
        A[i] += q * C[i];
    }
}

void two_scale_kernel_fused(double* A, double p,double *B, double q, double* C, uint64_t index_region){
    for(uint64_t i = 0; i < index_region; ++i){
        A[i] = p * B[i] + q * C[i];
    }
    // for(uint64_t i = 0; i < index_region; ++i){
    //     A[i] = p * B[i];
    // }
    // for(uint64_t i = 0; i < index_region; ++i){
    //     A[i] += q * C[i];
    // }
}

double two_scale_test(
    uint64_t index_region,
    uint64_t flops_per_test,
    FILE* output_file)
{
    uint64_t repeat_count = flops_per_test / index_region;

    double *A, *B, *C;
    double p = 3.0;
    double q = 2.0;
    data_init(&A, &B, &C, index_region);
    
    // warm up
#if defined(TWO_SCALE_NAIVE)
    two_scale_kernel_naive(A, p, B, q, C, index_region);
#elif defined(TWO_SCALE_FUSED)
    two_scale_kernel_fused(A, p, B, q, C, index_region);
#else
    fprintf(stderr, "no define test kernel !!!");
    fflush(stderr);
#endif
    // test
    double time_start = dtime();
    uint64_t cycle_start = rdtsc();
    for(int r = 0; r < repeat_count; r++){
#if defined(TWO_SCALE_NAIVE)
        two_scale_kernel_naive(A, p, B, q, C, index_region);
#elif defined(TWO_SCALE_FUSED)
        two_scale_kernel_fused(A, p, B, q, C, index_region);
#else
        fprintf(stderr, "no define test kernel !!!");
        fflush(stderr);
#endif
    }
    uint64_t cycle_end = rdtsc();
    double time_end = dtime();
    uint64_t cycles = cycle_end - cycle_start;
    double time = time_end - time_start;

    double MFLOPS =  3. * repeat_count * index_region / 1e6 / time;
    double MBPS = 3. * 8. * repeat_count * index_region / 1e6 / time;

    printf("cycle : %ld\n", cycles);
    printf("time : %e\n", time);
    printf("frequent : %.2lf GHz\n", cycles / 1e9 / time);
    
    printf("repeat count : %ld\n", repeat_count);
    printf("performance : %8.4lf MFLOPS\n", MFLOPS);
    printf("bandwidth : %8.4lf MB/s\n", MBPS);
    fflush(stdout);

    fprintf(output_file, "%ld %8.4lf\n", index_region, MFLOPS);
    fflush(output_file);

    double sum = 0.;
    for(uint64_t i = 0; i < index_region; ++i){
        sum += A[i];
    }

    data_release(A, B, C);
    return sum;
}

int main(){
    uint64_t index_region_start = env_get_uint64("INDEX_REGION_START", 256);
    uint64_t index_region_end = env_get_uint64("INDEX_REGION_END", 1073741824);
    uint64_t sample_points = env_get_uint64("SAMPLE_POINTS", 4);         
    uint64_t flops_per_test = env_get_uint64("FLOPS_PER_TEST", index_region_end); 


    const char* latency_output_filename_dir = env_get_string("LATENCY_OUTPUT_FILENAME_DIR", "./data/"); 
    const char* latency_output_filename_prefix = env_get_string("LATENCY_OUTPUT_FILENAME_PREFIX", "two_scale_mflops"); 
    const char* latency_output_filename_name = env_get_string("LATENCY_OUTPUT_FILENAME_NAME", "naive"); 
    const char* latency_output_filename_suffix = env_get_string("LATENCY_OUTPUT_FILENAME_SUFFIX", ".dat"); 

    printf("index_region_start : %ld\n", index_region_start);
    printf("index_region_end : %ld\n", index_region_end);
    printf("sample_points : %ld\n", sample_points);
    printf("flops_per_test : %ld\n", flops_per_test);
    printf("latency_output_filename_DIR : %s\n", latency_output_filename_dir);
    printf("latency_output_filename_prefix : %s\n", latency_output_filename_prefix);
    printf("latency_output_filename_name : %s\n", latency_output_filename_name);
    printf("latency_output_filename_suffix : %s\n", latency_output_filename_suffix);
    fflush(stdout);

    char filename[128];
    sprintf(filename, "%s%s_%s%s", latency_output_filename_dir, latency_output_filename_prefix,latency_output_filename_name, latency_output_filename_suffix);
    FILE* output_file = fopen(filename, "w");

    for(uint64_t index_region_pow_2 = index_region_start; index_region_pow_2 < index_region_end; index_region_pow_2 *= 2){
        uint64_t step = index_region_pow_2 / sample_points;
        for(uint64_t index_region = index_region_pow_2; index_region < min(index_region_pow_2 * 2, index_region_end); index_region += step){
            two_scale_test(index_region, flops_per_test, output_file);
        }
    }
    two_scale_test(index_region_end, flops_per_test, output_file);

    fclose(output_file);
}