#include "mem.h"

void data_access_test(
    uint64_t access_region, 
    uint64_t access_count, 
    uint64_t prefetch_count,
    uint64_t chains, 
    FILE* cycle_file)
{
    printf("--------------------------------------------------------------\n");
    uint64_t repeat_count = max(access_count / access_region, 1);
    // uint64_t repeat_count = 1;

    uint64_t index_region =  access_region / CACHE_LINE_SIZE;
 
    assert(index_region != -1);

    uint64_t** ptrs;
    gen_access_list_multichain(&ptrs, index_region, chains);

    printf("cycle size : %ld\n", ptr_list_cycle_detect(ptrs[0], index_region));

    Node_t** node_list_list;
    gen_node_list_multichain(&node_list_list, ptrs, index_region, prefetch_count, chains);

#ifdef DEF_PTRCHASE
    memory_test_kernel_ptrchase_multichain(node_list_list, index_region, repeat_count, chains, cycle_file);
#else
#ifdef DEF_RANDOM_WITHOUT_PTRCHASE
    memory_test_kernel_random_without_ptrchase(ptrs, access_region, index_region, repeat_count, cycle_file);
#endif
#ifdef DEF_SEQENTIAL_WITHOUT_PTRCHASE
    memory_test_kernel_seqential_without_ptrchase(access_region, index_region, repeat_count, cycle_file);
#endif
#endif

    release_access_list_multichain(ptrs, chains);

    release_node_list_multichain(node_list_list, chains);
    printf("--------------------------------------------------------------\n");
}

int main(){

    uint64_t sample_points = env_get_uint64("SAMPLE_POINTS", 4);         
    uint64_t access_region_start = env_get_uint64("ACCESS_REGION_START", 256);  // bytes >= 256
    // uint64_t access_region_end = env_get_uint64("ACCESS_REGION_END", 256);  // bytes
    uint64_t access_region_end = env_get_uint64("ACCESS_REGION_END", 2147483648);  // bytes
    uint64_t prefetch_count = env_get_uint64("PREFETCH_COUNT", 0);  // bytes
    
    uint64_t access_count = env_get_uint64("ACCESS_COUNT", access_region_end * 2);  // data accessed larger than L3 cache size

    const char* latency_output_filename_dir = env_get_string("LATENCY_OUTPUT_FILENAME_DIR", "./data/"); 
    const char* latency_output_filename_prefix = env_get_string("LATENCY_OUTPUT_FILENAME_PREFIX", "mem"); 
    const char* latency_output_filename_suffix = env_get_string("LATENCY_OUTPUT_FILENAME_SUFFIX", ".dat"); 

#ifndef DEF_CHAIN_COUNT
    fprintf(stderr, "DEF_CHAIN_COUNT is undefined when compling\n");
    fprintf(stderr, "must define DEF_CHAIN_COUNT == chains\n");
    fflush(stderr);
    abort();
#endif

    uint64_t chains = DEF_CHAIN_COUNT;

    printf("access_count : %ld\n", access_count);
    printf("sample_points : %ld\n", sample_points);
    printf("access_region_start : %ld\n", access_region_start);
    printf("access_region_end : %ld\n", access_region_end);
    printf("chains : %ld\n", chains);
#ifdef DEF_PREFETCH
    printf("prefetch_count : %ld\n", prefetch_count);
#endif
    printf("sizeof Node_t : %ld byte\n", sizeof(Node_t));
#ifdef _OPENMP
    printf("number of threads : %ld\n", omp_get_max_threads());
#endif
    printf("latency_output_filename_DIR : %s\n", latency_output_filename_dir);
    printf("latency_output_filename_prefix : %s\n", latency_output_filename_prefix);
    printf("latency_output_filename_suffix : %s\n", latency_output_filename_suffix);
    fflush(stdout);

    char filename[128];
    sprintf(filename, "%s%s%s", latency_output_filename_dir, latency_output_filename_prefix, latency_output_filename_suffix);
    FILE* cycle_file = fopen(filename, "w");

    for(uint64_t access_region_pow_2 = access_region_start; access_region_pow_2 < access_region_end; access_region_pow_2 *= 2){
        uint64_t step = access_region_pow_2 / sample_points;
        for(uint64_t access_region = access_region_pow_2; access_region < min(access_region_pow_2 * 2, access_region_end); access_region += step){
            data_access_test(access_region, access_count, prefetch_count,chains, cycle_file);
        }
    }

    data_access_test(access_region_end, access_count, prefetch_count, chains, cycle_file);
    
    fclose(cycle_file);

    return 0;
}