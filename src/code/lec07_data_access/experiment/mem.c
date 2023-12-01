#include "mem.h"


uint64_t memory_test_kernel_ptrchase(
    Node_t* node_list,
    uint64_t index_region,
    uint64_t repeat_count,
    FILE* cycle_file
){
    printf("In memory_test_kernel_ptrchase\n");
    printf("index_region : %ld\n", index_region);
    printf("repeat_count : %ld\n", repeat_count);
    fflush(stdout);

    // warm up
    Node_t* pre_ptr = &node_list[0];
    
    // travel random node list
    ...
    
    
    double time_start = dtime();
    uint64_t cycle_start = rdtsc();

    
    for(int i = 0; i < repeat_count; i++){
        // travel random node list
        ...
    }

    uint64_t cycle_end = rdtsc();
    double time_end = dtime();

    uint64_t cycle = cycle_end - cycle_start;
    double time = time_end - time_start;

    double cycles = cycle * 1.0 / index_region / repeat_count;

    uint64_t access_region = index_region * sizeof(Node_t);

    double GBPS1 = 1.0 * repeat_count * index_region * sizeof(Node_t) / 1024./ 1024./ 1024./ time;

    printf("time : %.2lf s\n", time);
    printf("frequent : %.2lf GHz\n", cycles / 1e9 / time);
    printf("cycles : %8.4lf \n", cycles);
    printf("GBPS1 : %8.4lf GB/s\n", GBPS1);
    // printf("GBPS2 : %8.4lf GB/s\n", GBPS2);
    fflush(stdout);

    fprintf(cycle_file, "%ld %8.4lf %8.4lf\n", access_region, cycles, GBPS1);
    fflush(cycle_file);

    return (uint64_t)pre_ptr;
}

void data_access_test(
    uint64_t access_region, 
    uint64_t access_count, 
    FILE* cycle_file)
{
    uint64_t repeat_count = max(access_count / access_region, 1);

    uint64_t index_region =  access_region / CACHE_LINE_SIZE;
 
    assert(index_region != -1);

    uint64_t* ptr;
    gen_random_list(&ptr, index_region);

    Node_t* node_list;
    gen_node_list(&node_list, ptr, index_region);

    memory_test_kernel_ptrchase(node_list, index_region, repeat_count, cycle_file);

    release_access_list(ptr);

    release_node_list(node_list);
}

int main(){

    uint64_t sample_points = env_get_uint64("SAMPLE_POINTS", ...);         
    uint64_t access_region_start = env_get_uint64("ACCESS_REGION_START", ...);  // bytes >= 256
    uint64_t access_region_end = env_get_uint64("ACCESS_REGION_END", ...);  // bytes
    
    uint64_t access_count = env_get_uint64("ACCESS_COUNT", ...);  // data accessed larger than L3 cache size

    const char* latency_output_filename_dir = env_get_string("LATENCY_OUTPUT_FILENAME_DIR", "./data/"); 
    const char* latency_output_filename_prefix = env_get_string("LATENCY_OUTPUT_FILENAME_PREFIX", "mem"); 
    const char* latency_output_filename_name = env_get_string("LATENCY_OUTPUT_FILENAME_NAME", "naive"); 
    const char* latency_output_filename_suffix = env_get_string("LATENCY_OUTPUT_FILENAME_SUFFIX", ".dat"); 

    printf("access_count : %ld\n", access_count);
    printf("sample_points : %ld\n", sample_points);
    printf("access_region_start : %ld\n", access_region_start);
    printf("access_region_end : %ld\n", access_region_end);

    printf("sizeof Node_t : %ld byte\n", sizeof(Node_t));

    printf("latency_output_filename_DIR : %s\n", latency_output_filename_dir);
    printf("latency_output_filename_prefix : %s\n", latency_output_filename_prefix);
    printf("latency_output_filename_suffix : %s\n", latency_output_filename_suffix);
    fflush(stdout);

    char filename[128];
    sprintf(filename, "%s%s_%s%s", latency_output_filename_dir, latency_output_filename_prefix, latency_output_filename_name, latency_output_filename_suffix);
    FILE* cycle_file = fopen(filename, "w");

    for(uint64_t access_region_pow_2 = access_region_start; access_region_pow_2 < access_region_end; access_region_pow_2 *= 2){
        uint64_t step = access_region_pow_2 / sample_points;
        for(uint64_t access_region = access_region_pow_2; access_region < min(access_region_pow_2 * 2, access_region_end); access_region += step){
            data_access_test(access_region, access_count, cycle_file);
        }
    }

    data_access_test(access_region_end, access_count, cycle_file);
    
    fclose(cycle_file);

    return 0;
}