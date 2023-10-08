#include "util.h"

#include <immintrin.h>


void data_init(double** A_p, double** B_p, double** C_p, double** D_p, uint64_t len){
    *A_p = (double*)aligned_alloc(64, len * sizeof(double));
    *B_p = (double*)aligned_alloc(64, len * sizeof(double));
    *C_p = (double*)aligned_alloc(64, len * sizeof(double));
    *D_p = (double*)aligned_alloc(64, len * sizeof(double));
    
    double* A = *A_p;
    double* B = *B_p;
    double* C = *C_p;
    double* D = *D_p;

    for(uint64_t i = 0; i < len; ++i){
        A[i] = 1.;
        B[i] = 1.;
        C[i] = 1.;
        D[i] = 1.;
    }
}

void data_release(double* A, double* B, double* C, double* D){
    free(A);
    free(B);
    free(C);
    free(D);
}

double vector_triads_kernel_naive(double* A, double *B, double* C, double *D, uint64_t index_region, uint64_t repeat_count, FILE* output_file){
    // warm up
    for(uint64_t i = 0; i < index_region; ++i){
        A[i] = B[i] + C[i] * D[i];
    }

    // test
    double time_start = dtime();
    uint64_t cycle_start = rdtsc();
    for(int r = 0; r < repeat_count; r++){
        for(uint64_t i = 0; i < index_region; ++i){
            A[i] = B[i] + C[i] * D[i];
        }
    }

    uint64_t cycle_end = rdtsc();
    double time_end = dtime();
    uint64_t cycles = cycle_end - cycle_start;
    double time = time_end - time_start;

    double MFLOPS =  2. * repeat_count * index_region / 1e6 / time;
    double MBPS = 4. * 8. * repeat_count * index_region / 1e6 / time;

    printf("cycle : %ld\n", cycles);
    printf("time : %e\n", time);
    printf("frequent : %.2lf GHz\n", cycles / 1e9 / time);
    
    printf("repeat count : %ld\n", repeat_count);
    printf("performance : %8.4lf MFLOPS\n", MFLOPS);
    printf("bandwidth : %8.4lf MB/s\n", MBPS);
    fflush(stdout);

    fprintf(output_file, "%ld %8.4lf\n", index_region, MFLOPS);
    fflush(output_file);

    // cheat compiler
    double sum = 0.;
    for(uint64_t i = 0; i < index_region; ++i){
        sum += A[i];
    }
    return sum;
}

double vector_triads_kernel_model(uint64_t index_region, FILE* output_file){
    double MFLOPS;
    if(index_region <= (L1_CACHE_SIZE / 4 / sizeof(double))){
        MFLOPS = 15467.;   // L1
    }else if(index_region <= (L2_CACHE_SIZE / 4 / sizeof(double))){
        MFLOPS = 5880.;    // L2
    }else if(index_region <= (L3_CACHE_SIZE / 4 / sizeof(double))){
        MFLOPS = 2578.;    // L3
    }else{
        MFLOPS = 1132.;    // Mem
    }
    fprintf(output_file, "%ld %8.4lf\n", index_region, MFLOPS);
    fflush(output_file);
    return 0.;
}

double vector_triads_kernel_avx512(double* A, double *B, double* C, double *D, uint64_t index_region, uint64_t repeat_count, FILE* output_file){
    index_region = index_region / 8 * 8;
    fflush(stdout);
    assert(index_region % 8 == 0);
    // warm up
    for(uint64_t i = 0; i < index_region; i += 8){
        __m512d vB = _mm512_load_pd(B + i);
        __m512d vC = _mm512_load_pd(C + i);
        __m512d vD = _mm512_load_pd(D + i);
        __m512d vA = _mm512_fmadd_pd(vC, vD, vB);
        _mm512_store_pd(A + i, vA);
    }

    // test
    double time_start = dtime();
    uint64_t cycle_start = rdtsc();
    for(int r = 0; r < repeat_count; r++){
        for(uint64_t i = 0; i < index_region; i += 8){
            __m512d vB = _mm512_load_pd(B + i);
            __m512d vC = _mm512_load_pd(C + i);
            __m512d vD = _mm512_load_pd(D + i);
            __m512d vA = _mm512_fmadd_pd(vC, vD, vB);
            _mm512_store_pd(A + i, vA);
        }
    }

    uint64_t cycle_end = rdtsc();
    double time_end = dtime();
    uint64_t cycles = cycle_end - cycle_start;
    double time = time_end - time_start;

    double MFLOPS =  2. * repeat_count * index_region / 1e6 / time;
    double MBPS = 4. * 8. * repeat_count * index_region / 1e6 / time;

    printf("cycle : %ld\n", cycles);
    printf("time : %e\n", time);
    printf("frequent : %.2lf GHz\n", cycles / 1e9 / time);
    
    printf("repeat count : %ld\n", repeat_count);
    printf("performance : %8.4lf MFLOPS\n", MFLOPS);
    printf("bandwidth : %8.4lf MB/s\n", MBPS);
    fflush(stdout);

    fprintf(output_file, "%ld %8.4lf\n", index_region, MFLOPS);
    fflush(output_file);

    // cheat compiler
    double sum = 0.;
    for(uint64_t i = 0; i < index_region; ++i){
        sum += A[i];
    }
    return sum;
}

double vector_triads_kernel_avx512_nt(double* A, double *B, double* C, double *D, uint64_t index_region, uint64_t repeat_count, FILE* output_file){
    index_region = index_region / 8 * 8;
    fflush(stdout);
    assert(index_region % 8 == 0);
    // warm up
    for(uint64_t i = 0; i < index_region; i += 8){
        __m512d vB = _mm512_load_pd(B + i);
        __m512d vC = _mm512_load_pd(C + i);
        __m512d vD = _mm512_load_pd(D + i);
        __m512d vA = _mm512_fmadd_pd(vC, vD, vB);
        _mm512_stream_pd(A + i, vA);
    }

    // test
    double time_start = dtime();
    uint64_t cycle_start = rdtsc();
    for(int r = 0; r < repeat_count; r++){
        for(uint64_t i = 0; i < index_region; i += 8){
            __m512d vB = _mm512_load_pd(B + i);
            __m512d vC = _mm512_load_pd(C + i);
            __m512d vD = _mm512_load_pd(D + i);
            __m512d vA = _mm512_fmadd_pd(vC, vD, vB);
            _mm512_stream_pd(A + i, vA);
        }
    }

    uint64_t cycle_end = rdtsc();
    double time_end = dtime();
    uint64_t cycles = cycle_end - cycle_start;
    double time = time_end - time_start;

    double MFLOPS =  2. * repeat_count * index_region / 1e6 / time;
    double MBPS = 4. * 8. * repeat_count * index_region / 1e6 / time;

    printf("cycle : %ld\n", cycles);
    printf("time : %e\n", time);
    printf("frequent : %.2lf GHz\n", cycles / 1e9 / time);
    
    printf("repeat count : %ld\n", repeat_count);
    printf("performance : %8.4lf MFLOPS\n", MFLOPS);
    printf("bandwidth : %8.4lf MB/s\n", MBPS);
    fflush(stdout);

    fprintf(output_file, "%ld %8.4lf\n", index_region, MFLOPS);
    fflush(output_file);

    // cheat compiler
    double sum = 0.;
    for(uint64_t i = 0; i < index_region; ++i){
        sum += A[i];
    }
    return sum;
}

void vector_triads_test(
    uint64_t index_region,
    uint64_t flops_per_test,
    FILE* output_file)
{
    uint64_t adapative_repeat_count = flops_per_test / index_region;

    double *A, *B, *C, *D;
    data_init(&A, &B, &C, &D, index_region);

#if defined(VECTOR_TRIADS_NAIVE)
    vector_triads_kernel_naive(A, B, C, D, index_region, adapative_repeat_count, output_file);
#elif defined(VECTOR_TRIADS_AVX512)
    vector_triads_kernel_avx512(A, B, C, D, index_region, adapative_repeat_count, output_file);
#elif defined(VECTOR_TRIADS_AVX512_NT)
    vector_triads_kernel_avx512_nt(A, B, C, D, index_region, adapative_repeat_count, output_file);
#elif defined(VECTOR_TRIADS_MODEL)
    vector_triads_kernel_model(index_region, output_file);
#else

    fprintf(stderr, "no define test kernel !!!");
    fflush(stderr);
#endif

    data_release(A, B, C, D);
    return;
}

int main(){
    uint64_t index_region_start = env_get_uint64("INDEX_REGION_START", 32);
    uint64_t index_region_end = env_get_uint64("INDEX_REGION_END", 134217728);
    uint64_t sample_points = env_get_uint64("SAMPLE_POINTS", 4);         
    uint64_t flops_per_test = env_get_uint64("FLOPS_PER_TEST", 1073741824); 


    const char* latency_output_filename_dir = env_get_string("LATENCY_OUTPUT_FILENAME_DIR", "./data/"); 
    const char* latency_output_filename_prefix = env_get_string("LATENCY_OUTPUT_FILENAME_PREFIX", "triads_mflops"); 
    const char* latency_output_filename_suffix = env_get_string("LATENCY_OUTPUT_FILENAME_SUFFIX", ".dat"); 

    printf("index_region_start : %ld\n", index_region_start);
    printf("index_region_end : %ld\n", index_region_end);
    printf("sample_points : %ld\n", sample_points);
    printf("flops_per_test : %ld\n", flops_per_test);
    printf("latency_output_filename_DIR : %s\n", latency_output_filename_dir);
    printf("latency_output_filename_prefix : %s\n", latency_output_filename_prefix);
    printf("latency_output_filename_suffix : %s\n", latency_output_filename_suffix);
    fflush(stdout);

    char filename[128];
    sprintf(filename, "%s%s%s", latency_output_filename_dir, latency_output_filename_prefix, latency_output_filename_suffix);
    FILE* output_file = fopen(filename, "w");

    for(uint64_t index_region_pow_2 = index_region_start; index_region_pow_2 < index_region_end; index_region_pow_2 *= 2){
        uint64_t step = index_region_pow_2 / sample_points;
        for(uint64_t index_region = index_region_pow_2; index_region < min(index_region_pow_2 * 2, index_region_end); index_region += step){
            vector_triads_test(index_region, flops_per_test, output_file);
        }
    }
    vector_triads_test(index_region_end, flops_per_test, output_file);

    fclose(output_file);
}