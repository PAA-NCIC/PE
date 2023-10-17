#include "util.h"

#include <papi.h>

void check_(int ret, const char * fun) {
    if (ret != PAPI_OK){ 
        fprintf(stderr, "%s failed (%s)\n", fun, PAPI_strerror(ret)); 
        exit(1); 
    } 
} 

#define check(f) check_(f, #f)

#define index2(r,c,ld) ((c) + (r) * (ld))

void matrix_init(uint64_t row, uint64_t col, double** D_p, uint64_t ld, double init_value){
    if(ld < col){
        fprintf(stderr, "ld is invalid in matrix init !!!\n");
        exit(-1);
    }
    *D_p = (double*)aligned_alloc(64, row * ld * sizeof(double));
    double* D = *D_p;
    for(uint64_t r = 0; r < row; ++r){
        for(uint64_t c = 0; c < col; ++c){
            D[index2(r,c,ld)] = init_value;
        }
    } 
}

void data_init(uint64_t m, uint64_t n, uint64_t k, double** A_p, uint64_t lda,double** B_p, uint64_t ldb, double** C_p, uint64_t ldc){
    matrix_init(m, k, A_p, lda, 1.);
    matrix_init(k, n, B_p, ldb, 1.);
    matrix_init(m, n, C_p, ldc, 0.);
}

void data_release(double* A, double* B, double* C){
    free(A);
    free(B);
    free(C);
}

void dgemm_kernel_naive(uint64_t M, uint64_t N, uint64_t K, double* A, uint64_t lda, double* B, uint64_t ldb, double* C, uint64_t ldc){
    for(uint64_t i = 0; i < M; ++i){
        for(uint64_t j = 0; j < N; ++j){
            for(uint64_t k = 0; k < K; ++k){
                C[index2(i, j, ldc)] += A[index2(i, k, lda)] * B[index2(k, j, ldb)];
            }
        }
    }
}

void dgemm_kernel_recursively_blocked(uint64_t M, uint64_t N, uint64_t K, double* A, uint64_t lda, double* B, uint64_t ldb, double* C, uint64_t ldc){
    const uint64_t BLOCK_SIZE = env_get_uint64("BLOCK_SIZE", 128);  // bytes >= 256
    if(M == N && N == K && M <= BLOCK_SIZE){
        dgemm_kernel_naive(M, N, K, A, lda, B, ldb, C, ldc);
        return;
    }else if(max3(M, N, K) == M){
        dgemm_kernel_recursively_blocked(M / 2, N, K, A, lda, B, ldb, C, ldc);
        dgemm_kernel_recursively_blocked(M / 2, N, K, &A[index2(M/2, 0, lda)], lda, B, ldb, &C[index2(M/2, 0, ldc)], ldc);
        return;
    }else if(max3(M, N, K) == N){
        dgemm_kernel_recursively_blocked(M, N / 2, K, A, lda, B, ldb, C, ldc);
        dgemm_kernel_recursively_blocked(M, N / 2, K, A, lda, &B[index2(0, N/2, ldb)], ldb, &C[index2(0, N/2, ldc)], ldc); 
        return;
    }else if(max3(M, N, K) == K){
        dgemm_kernel_recursively_blocked(M, N, K / 2, A, lda, B, ldb, C, ldc);
        dgemm_kernel_recursively_blocked(M, N, K / 2, &A[index2(0, K/2, lda)], lda, &B[index2(K/2, 0, ldb)], ldb, C, ldc); 
        return;
    }else{
        assert(false);
    }
}

double dgemm_kernel_launcher(uint64_t M, uint64_t N, uint64_t K, double* A, uint64_t lda, double* B, uint64_t ldb, double* C, uint64_t ldc, uint64_t repeat_count, FILE* output_file){
    // warm up
#if defined(DGEMM_NAIVE)
    dgemm_kernel_naive(M, N, K, A, lda, B, ldb, C, ldc);
#elif defined(DGEMM_RECURSIVELY_BLOCKED)
    dgemm_kernel_recursively_blocked(M, N, K, A, lda, B, ldb, C, ldc);
#else
    fprintf(stderr, "no define test kernel !!!\n");
    fflush(stderr);
    exit(-1);
#endif
    int event_set = PAPI_NULL;
    check(PAPI_create_eventset(&event_set));
    check(PAPI_add_named_event(event_set, "PAPI_L1_DCM"));
    check(PAPI_add_named_event(event_set, "PAPI_L2_DCM"));
    check(PAPI_add_named_event(event_set, "PAPI_L2_DCA"));
    check(PAPI_add_named_event(event_set, "PAPI_L3_TCM"));
    check(PAPI_add_named_event(event_set, "PAPI_L3_DCA"));
    check(PAPI_add_named_event(event_set, "PAPI_L3_ICA"));
    check(PAPI_add_named_event(event_set, "PAPI_L3_TCA"));
    check(PAPI_start(event_set));

    double time_start = dtime();
    uint64_t cycle_start = rdtsc();

    for(uint64_t repeat = 0; repeat < repeat_count; ++repeat){
#if defined(DGEMM_NAIVE)
        dgemm_kernel_naive(M, N, K, A, lda, B, ldb, C, ldc);
#elif defined(DGEMM_RECURSIVELY_BLOCKED)
        dgemm_kernel_recursively_blocked(M, N, K, A, lda, B, ldb, C, ldc);
#else
        fprintf(stderr, "no define test kernel !!!\n");
        fflush(stderr);
        exit(-1);
#endif
    }

    uint64_t cycle_end = rdtsc();
    double time_end = dtime();
    
    long long values[4];

    check(PAPI_stop(event_set, values));
    check(PAPI_cleanup_eventset(event_set));

    long long L1_data_cache_miss = values[0];
    long long L2_data_cache_miss = values[1];
    long long L2_data_cache_access = values[2];
    long long L3_cache_miss = values[3];
    long long L3_data_cache_access = values[4];
    long long L3_instruction_cache_access = values[5];
    long long L3_total_cache_access = values[6];

    double time = time_end - time_start;
    uint64_t cycles = cycle_end - cycle_start;

    double MFLOPS =  2. * repeat_count * M * N * K / 1e6 / time;
    uint64_t total_access = 4 * repeat_count * M * N * K;
    double MBPS = 8. * total_access / 1e6 / time;
    
    printf("cycle : %ld\n", cycles);
    printf("time : %e\n", time);
    printf("frequent : %.2lf GHz\n", cycles / 1e9 / time);
    
    printf("repeat count : %ld\n", repeat_count);
    printf("performance : %8.4lf MFLOPS\n", MFLOPS);
    printf("bandwidth : %8.4lf MB/s\n", MBPS);
    printf("Level 1 data cache misses : %ld\n", L1_data_cache_miss);
    printf("Level 2 data cache misses : %ld\n", L2_data_cache_miss);
    printf("Level 2 data cache accesses : %ld\n", L2_data_cache_access);
    printf("Level 3 cache misses : %ld\n", L3_cache_miss);
    printf("Level 3 data cache accesses : %ld\n", L3_data_cache_access);
    printf("Level 3 instruction cache accesses : %ld\n", L3_instruction_cache_access);
    printf("Level 3 total cache accesses : %ld\n", L3_total_cache_access);
    printf("Total data access : %ld\n", total_access);
    printf("L1 miss rate : %8.4lf %%\n", 100. * L1_data_cache_miss / total_access);
    printf("L2 miss rate : %8.4lf %%\n", 100. * L2_data_cache_miss / total_access);
    printf("L3 miss rate : %8.4lf %%\n", 100. * L3_cache_miss / total_access);
    fflush(stdout);

    fprintf(output_file, "%ld %8.4lf %8.7lf %8.7lf %8.7lf\n", M, MFLOPS, L1_data_cache_miss * 1.0 / total_access, L2_data_cache_miss * 1.0 / total_access, L3_cache_miss * 1.0 / total_access);
    fflush(output_file);
    
    // cheat compiler
    double sum = 0.;
    for(uint64_t r = 0; r < M; ++r){
        for(uint64_t c = 0; c < N; ++c){
            sum += C[index2(r, c, ldc)];
        }
    }
    return sum;
}

void dgemm_check(uint64_t m, uint64_t n, uint64_t k){
    double *A, *B, *C, *C_answer;
    uint64_t lda = k;
    uint64_t ldb = n;
    uint64_t ldc = n;
    matrix_init(m, k, &A, lda, 1.);
    matrix_init(k, n, &B, ldb, 1.);
    matrix_init(m, n, &C, ldc, 0.);
    matrix_init(m, n, &C_answer, ldc, 0.);

    dgemm_kernel_naive(m, n, k, A, lda, B, ldb, C_answer, ldc);

    dgemm_kernel_recursively_blocked(m, n, k, A, lda, B, ldb, C, ldc);

    double max_diff = 0.;
    for(uint64_t r = 0; r < m; ++r){
        for(uint64_t c = 0; c < n; ++c){
            double diff = fabs(C[index2(r,c,ldc)] - C_answer[index2(r,c,ldc)]);
            max_diff = max(diff, max_diff);
        }
    }
    printf("max_diff : %lf\n", max_diff);
    fflush(stdout);

    free(A);
    free(B);
    free(C);
    free(C_answer);
}

void dgemm_test(
    uint64_t m,
    uint64_t n,
    uint64_t k,
    uint64_t flops_per_test,
    FILE* output_file)
{
    // dgemm_check(m, n, k);

    uint64_t adapative_repeat_count = max(flops_per_test / m / n / k / 256,1);
    // uint64_t adapative_repeat_count = 1;

    double *A, *B, *C;
    uint64_t lda = k;
    uint64_t ldb = n;
    uint64_t ldc = n;
    data_init(m, n, k, &A, lda, &B, ldb, &C, ldc);

    dgemm_kernel_launcher(m, n, k, A, lda, B, ldb, C, ldc, adapative_repeat_count, output_file);

    data_release(A, B, C);
    return;
}


int main(){
    int ver = PAPI_library_init(PAPI_VER_CURRENT);
    if (ver != PAPI_VER_CURRENT){ 
        fprintf(stderr, "PAPI_library_init(%d) failed (returned %d)\n", PAPI_VER_CURRENT, ver); 
        exit(1); 
    }

    uint64_t mnk_region_start = env_get_uint64("MNK_REGION_START", 128);
    uint64_t mnk_region_end = env_get_uint64("MNK_REGION_END", 4096);
    uint64_t sample_points = env_get_uint64("SAMPLE_POINTS", 16);  
    uint64_t sample_step = env_get_uint64("SAMPLE_STEP", 16);  
    uint64_t flops_per_test = env_get_uint64("FLOPS_PER_TEST", mnk_region_end * mnk_region_end * mnk_region_end); 

    const char* latency_output_filename_dir = env_get_string("LATENCY_OUTPUT_FILENAME_DIR", "./data/"); 
    const char* latency_output_filename_prefix = env_get_string("LATENCY_OUTPUT_FILENAME_PREFIX", "dgemm"); 
    const char* latency_output_filename_name = env_get_string("LATENCY_OUTPUT_FILENAME_NAME", "naive"); 
    const char* latency_output_filename_suffix = env_get_string("LATENCY_OUTPUT_FILENAME_SUFFIX", ".dat"); 

    printf("mnk_region_start : %ld\n", mnk_region_start);
    printf("mnk_region_end : %ld\n", mnk_region_end);
    printf("sample_points : %ld\n", sample_points);
    printf("flops_per_test : %ld\n", flops_per_test);
    printf("latency_output_filename_DIR : %s\n", latency_output_filename_dir);
    printf("latency_output_filename_prefix : %s\n", latency_output_filename_prefix);
    printf("latency_output_filename_name : %s\n", latency_output_filename_name);
    printf("latency_output_filename_suffix : %s\n", latency_output_filename_suffix);
    fflush(stdout);

    char filename[128];
    sprintf(filename, "%s%s_%s%s", latency_output_filename_dir, latency_output_filename_prefix, latency_output_filename_name, latency_output_filename_suffix);
    printf("filename : %s\n", filename);
    FILE* output_file = fopen(filename, "w");

    // for(uint64_t mnk_region_pow_2 = mnk_region_start; mnk_region_pow_2 < mnk_region_end; mnk_region_pow_2 *= 2){
    //     uint64_t step = mnk_region_pow_2 / sample_points;
    //     for(uint64_t mnk = mnk_region_pow_2; mnk < min(mnk_region_pow_2 * 2, mnk_region_end); mnk += step){
    //         dgemm_test(mnk, mnk, mnk, flops_per_test, output_file);
    //     }
    // }

    for(uint64_t mnk = mnk_region_start; mnk < mnk_region_end; mnk += sample_step){
        dgemm_test(mnk, mnk, mnk, flops_per_test, output_file);
    }
    dgemm_test(mnk_region_end, mnk_region_end, mnk_region_end, flops_per_test, output_file);

    fclose(output_file);
}