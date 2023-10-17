#pragma once

#include "util.h"

#ifndef DEF_CHAIN_COUNT
#define DEF_CHAIN_COUNT 1
#endif

#if !defined(DEF_GEN_RANDOM_LIST) && !defined(DEF_GEN_SEQUENTIAL_LIST)
#define DEF_GEN_RANDOM_LIST
#endif

typedef void* pointer;

#define CACHE_LINE_SIZE 64

#define FREQUENCY_GHZ 2.9

typedef struct {
    pointer next;
    pointer prefetch;
    char padding[CACHE_LINE_SIZE - 2 * sizeof(pointer)];
} Node_t;

static void gen_random_list(uint64_t** ptr_p, uint64_t index_region){
    printf("index_region : %ld\n", index_region);
    fflush(stdout);
    *ptr_p = aligned_alloc(64, sizeof(uint64_t) * index_region);
    uint64_t* ptr = *ptr_p;
    bool* access = aligned_alloc(64, sizeof(bool) * index_region);
    srand(index_region);
    for(uint64_t i = 0; i < index_region; ++i){
        ptr[i] = i;
    }
    // shuffle
    for(uint64_t i = 0; i < index_region; ++i){
        uint64_t target = rand() % index_region;
        uint64_t tmp = ptr[i];
        ptr[i] = ptr[target];
        ptr[target] = tmp;
    }
    // let cycle_size == index_region
    uint64_t cycle_size = 0;
    uint64_t gen_count = -1;
    // while(cycle_size < (uint64_t)(index_region * 0.9)){
    while(cycle_size != index_region){
        gen_count += 1;
        for(uint64_t i = 0; i < index_region; ++i){
            access[i] = false;
        }
        uint64_t pre_idx = 0;
        for(cycle_size = 0; cycle_size < index_region; ++cycle_size){
            if(access[pre_idx]){
                // 找一个没有被访问过的值与ptr[pre_idx]交换位置
                uint64_t non_access_count = 0;
                for(uint64_t i = 0; i < index_region; ++i){
                    if(access[i] == false){
                        non_access_count += 1;
                    }
                }
                uint64_t target_in_non_access = rand() % non_access_count;
                uint64_t target = -1;
                uint64_t non_access_index = 0;
                for(uint64_t i = 0; i < index_region; ++i){
                    if(access[i] == false){
                        if(non_access_index == target_in_non_access){
                            target = i;
                            break;
                        }
                        non_access_index += 1;
                    }
                }
                uint64_t tmp = ptr[pre_idx];
                ptr[pre_idx] = ptr[target];
                ptr[target] = tmp;
                break;
            }else{
                access[pre_idx] = true;
                pre_idx = ptr[pre_idx];
            }
        }
        printf("cycle_size : %ld\n", cycle_size);
        fflush(stdout);
    }
    free(access);
    printf("gen count : %ld\n", gen_count);
    fflush(stdout);
}

static void gen_sequential_list(uint64_t** ptr_p, uint64_t index_region){
    printf("index_region : %ld\n", index_region);
    fflush(stdout);
    *ptr_p = aligned_alloc(64, sizeof(uint64_t) * index_region);
    uint64_t* ptr = *ptr_p;
    for(uint64_t i = 0; i < index_region; ++i){
        ptr[i] = i + 1;
    }
    ptr[index_region - 1] = 0;
}

static void release_access_list(uint64_t* ptr){
    free(ptr);
}

static void gen_access_list_multichain(uint64_t*** ptrs_p, uint64_t index_region, uint64_t chains){
    *ptrs_p = aligned_alloc(64, sizeof(uint64_t*) * chains);
    uint64_t** ptrs = *ptrs_p;
    for(uint64_t c = 0; c < chains; ++c){
#if defined(DEF_GEN_RANDOM_LIST) && defined(DEF_GEN_SEQUENTIAL_LIST)
        fprintf(stderr, "DEF_GEN_RANDOM_LIST and DEF_GEN_SEQUENTIAL_LIST can not defined at the same time \n");
        abort();
#endif
#ifdef DEF_GEN_RANDOM_LIST
        gen_random_list(&ptrs[c], index_region);
#endif
#ifdef DEF_GEN_SEQUENTIAL_LIST
        gen_sequential_list(&ptrs[c], index_region);
#endif
    }
}

static void release_access_list_multichain(uint64_t** ptrs, uint64_t chains){
    for(uint64_t ci = 0; ci < chains; ++ci){
        release_access_list(ptrs[ci]);
    }
    free(ptrs);
}

static void gen_node_list(Node_t** node_list_p, uint64_t* ptr, uint64_t index_region,uint64_t prefetch_count){
    *node_list_p = aligned_alloc(64, sizeof(Node_t) * index_region);
    Node_t* node_list = *node_list_p;
    for(uint64_t i = 0; i < index_region; ++i){
        node_list[i].next = &node_list[ptr[i]];
        uint64_t pre_idx = ptr[i];
        for(uint64_t p = 0; p < prefetch_count; ++p){
            pre_idx = ptr[pre_idx];
        }
        node_list[i].prefetch = &node_list[pre_idx];
    }
}

static void release_node_list(Node_t* node_list){
    free(node_list);
}

static void gen_node_list_multichain(Node_t*** node_list_list_p, uint64_t** ptrs, uint64_t index_region,uint64_t prefetch_count, uint64_t chains){
    *node_list_list_p = aligned_alloc(64, sizeof(Node_t*) * chains);
    Node_t** node_list_list = *node_list_list_p;
    for(uint64_t i = 0; i < chains; ++i){
        gen_node_list(&node_list_list[i], ptrs[i], index_region, prefetch_count);
    }
}

static void release_node_list_multichain(Node_t** node_list_list, uint64_t chains){
    for(uint64_t ci = 0; ci < chains; ++ci){
        release_node_list(node_list_list[ci]);
    }
    free(node_list_list);
}

#define DEFINE_GLOBAL(x) \
    uint64_t global_pre_ptr_##x = 0;

#define UPDATE_GLOBAL(x) \
    global_pre_ptr_##x += (uint64_t)pre_ptr_##x;

#define REDUCE_GLOBAL(x) \
    global_pre_ptr += global_pre_ptr_##x;

#define DEFINE_CHAIN(x) \
    Node_t* node_list_##x = node_list_list[x];\
    Node_t* pre_ptr_##x = &node_list_##x[0];

#define RESET_CHAIN(x) \
    pre_ptr_##x = &node_list_##x[0];

#ifndef DEF_PREFETCH
#define ACCESS_CHAIN(x) \
    pre_ptr_##x = (Node_t*)(pre_ptr_##x->next);
#else
#define ACCESS_CHAIN(x) \
    pre_ptr_##x = (Node_t*)(pre_ptr_##x->next);\
    __builtin_prefetch((Node_t*)(pre_ptr_##x->prefetch));
#endif

#if DEF_CHAIN_COUNT > 0
#define DEFINE_CHAIN_0 DEFINE_CHAIN(0)
#define RESET_CHAIN_0 RESET_CHAIN(0)
#define ACCESS_CHAIN_0 ACCESS_CHAIN(0)
#define DEFINE_GLOBAL_0 DEFINE_GLOBAL(0)
#define UPDATE_GLOBAL_0 UPDATE_GLOBAL(0)
#define REDUCE_GLOBAL_0 REDUCE_GLOBAL(0)
#else 
#define DEFINE_CHAIN_0 
#define RESET_CHAIN_0 
#define ACCESS_CHAIN_0 
#define DEFINE_GLOBAL_0
#define UPDATE_GLOBAL_0
#define REDUCE_GLOBAL_0
#endif

#if DEF_CHAIN_COUNT > 1
#define DEFINE_CHAIN_1 DEFINE_CHAIN(1)
#define RESET_CHAIN_1 RESET_CHAIN(1)
#define ACCESS_CHAIN_1 ACCESS_CHAIN(1)
#define DEFINE_GLOBAL_1 DEFINE_GLOBAL(1)
#define UPDATE_GLOBAL_1 UPDATE_GLOBAL(1)
#define REDUCE_GLOBAL_1 REDUCE_GLOBAL(1)
#else 
#define DEFINE_CHAIN_1
#define RESET_CHAIN_1
#define ACCESS_CHAIN_1 
#define DEFINE_GLOBAL_1
#define UPDATE_GLOBAL_1
#define REDUCE_GLOBAL_1
#endif

#if DEF_CHAIN_COUNT > 2
#define DEFINE_CHAIN_2 DEFINE_CHAIN(2)
#define RESET_CHAIN_2 RESET_CHAIN(2)
#define ACCESS_CHAIN_2 ACCESS_CHAIN(2)
#define DEFINE_GLOBAL_2 DEFINE_GLOBAL(2)
#define UPDATE_GLOBAL_2 UPDATE_GLOBAL(2)
#define REDUCE_GLOBAL_2 REDUCE_GLOBAL(2)
#else
#define DEFINE_CHAIN_2
#define RESET_CHAIN_2
#define ACCESS_CHAIN_2
#define DEFINE_GLOBAL_2
#define UPDATE_GLOBAL_2
#define REDUCE_GLOBAL_2
#endif

#if DEF_CHAIN_COUNT > 3
#define DEFINE_CHAIN_3 DEFINE_CHAIN(3)
#define RESET_CHAIN_3 RESET_CHAIN(3)
#define ACCESS_CHAIN_3 ACCESS_CHAIN(3)
#define DEFINE_GLOBAL_3 DEFINE_GLOBAL(3)
#define UPDATE_GLOBAL_3 UPDATE_GLOBAL(3)
#define REDUCE_GLOBAL_3 REDUCE_GLOBAL(3)
#else
#define DEFINE_CHAIN_3
#define RESET_CHAIN_3
#define ACCESS_CHAIN_3
#define DEFINE_GLOBAL_3
#define UPDATE_GLOBAL_3
#define REDUCE_GLOBAL_3
#endif

#if DEF_CHAIN_COUNT > 4
#define DEFINE_CHAIN_4 DEFINE_CHAIN(4)
#define RESET_CHAIN_4 RESET_CHAIN(4)
#define ACCESS_CHAIN_4 ACCESS_CHAIN(4)
#define DEFINE_GLOBAL_4 DEFINE_GLOBAL(4)
#define UPDATE_GLOBAL_4 UPDATE_GLOBAL(4)
#define REDUCE_GLOBAL_4 REDUCE_GLOBAL(4)
#else
#define DEFINE_CHAIN_4
#define RESET_CHAIN_4
#define ACCESS_CHAIN_4
#define DEFINE_GLOBAL_4
#define UPDATE_GLOBAL_4
#define REDUCE_GLOBAL_4
#endif

#if DEF_CHAIN_COUNT > 5
#define DEFINE_CHAIN_5 DEFINE_CHAIN(5)
#define RESET_CHAIN_5 RESET_CHAIN(5)
#define ACCESS_CHAIN_5 ACCESS_CHAIN(5)
#define DEFINE_GLOBAL_5 DEFINE_GLOBAL(5)
#define UPDATE_GLOBAL_5 UPDATE_GLOBAL(5)
#define REDUCE_GLOBAL_5 REDUCE_GLOBAL(5)
#else
#define DEFINE_CHAIN_5
#define RESET_CHAIN_5
#define ACCESS_CHAIN_5
#define DEFINE_GLOBAL_5
#define UPDATE_GLOBAL_5
#define REDUCE_GLOBAL_5
#endif

#if DEF_CHAIN_COUNT > 6
#define DEFINE_CHAIN_6 DEFINE_CHAIN(6)
#define RESET_CHAIN_6 RESET_CHAIN(6)
#define ACCESS_CHAIN_6 ACCESS_CHAIN(6)
#define DEFINE_GLOBAL_6 DEFINE_GLOBAL(6)
#define UPDATE_GLOBAL_6 UPDATE_GLOBAL(6)
#define REDUCE_GLOBAL_6 REDUCE_GLOBAL(6)
#else
#define DEFINE_CHAIN_6
#define RESET_CHAIN_6
#define ACCESS_CHAIN_6
#define DEFINE_GLOBAL_6
#define UPDATE_GLOBAL_6
#define REDUCE_GLOBAL_6
#endif

#if DEF_CHAIN_COUNT > 7
#define DEFINE_CHAIN_7 DEFINE_CHAIN(7)
#define RESET_CHAIN_7 RESET_CHAIN(7)
#define ACCESS_CHAIN_7 ACCESS_CHAIN(7)
#define DEFINE_GLOBAL_7 DEFINE_GLOBAL(7)
#define UPDATE_GLOBAL_7 UPDATE_GLOBAL(7)
#define REDUCE_GLOBAL_7 REDUCE_GLOBAL(7)
#else
#define DEFINE_CHAIN_7
#define RESET_CHAIN_7
#define ACCESS_CHAIN_7
#define DEFINE_GLOBAL_7
#define UPDATE_GLOBAL_7
#define REDUCE_GLOBAL_7
#endif

#if DEF_CHAIN_COUNT > 8
#define DEFINE_CHAIN_8 DEFINE_CHAIN(8)
#define RESET_CHAIN_8 RESET_CHAIN(8)
#define ACCESS_CHAIN_8 ACCESS_CHAIN(8)
#define DEFINE_GLOBAL_8 DEFINE_GLOBAL(8)
#define UPDATE_GLOBAL_8 UPDATE_GLOBAL(8)
#define REDUCE_GLOBAL_8 REDUCE_GLOBAL(8)
#else
#define DEFINE_CHAIN_8
#define RESET_CHAIN_8
#define ACCESS_CHAIN_8
#define DEFINE_GLOBAL_8
#define UPDATE_GLOBAL_8
#define REDUCE_GLOBAL_8
#endif

#if DEF_CHAIN_COUNT > 9
#define DEFINE_CHAIN_9 DEFINE_CHAIN(9)
#define RESET_CHAIN_9 RESET_CHAIN(9)
#define ACCESS_CHAIN_9 ACCESS_CHAIN(9)
#define DEFINE_GLOBAL_9 DEFINE_GLOBAL(9)
#define UPDATE_GLOBAL_9 UPDATE_GLOBAL(9)
#define REDUCE_GLOBAL_9 REDUCE_GLOBAL(9)
#else
#define DEFINE_CHAIN_9
#define RESET_CHAIN_9
#define ACCESS_CHAIN_9
#define DEFINE_GLOBAL_9
#define UPDATE_GLOBAL_9
#define REDUCE_GLOBAL_9
#endif

#if DEF_CHAIN_COUNT > 10
#define DEFINE_CHAIN_10 DEFINE_CHAIN(10)
#define RESET_CHAIN_10 RESET_CHAIN(10)
#define ACCESS_CHAIN_10 ACCESS_CHAIN(10)
#define DEFINE_GLOBAL_10 DEFINE_GLOBAL(10)
#define UPDATE_GLOBAL_10 UPDATE_GLOBAL(10)
#define REDUCE_GLOBAL_10 REDUCE_GLOBAL(10)
#else
#define DEFINE_CHAIN_10
#define RESET_CHAIN_10
#define ACCESS_CHAIN_10
#define DEFINE_GLOBAL_10
#define UPDATE_GLOBAL_10
#define REDUCE_GLOBAL_10
#endif

#if DEF_CHAIN_COUNT > 11
#define DEFINE_CHAIN_11 DEFINE_CHAIN(11)
#define RESET_CHAIN_11 RESET_CHAIN(11)
#define ACCESS_CHAIN_11 ACCESS_CHAIN(11)
#define DEFINE_GLOBAL_11 DEFINE_GLOBAL(11)
#define UPDATE_GLOBAL_11 UPDATE_GLOBAL(11)
#define REDUCE_GLOBAL_11 REDUCE_GLOBAL(11)
#else
#define DEFINE_CHAIN_11
#define RESET_CHAIN_11
#define ACCESS_CHAIN_11
#define DEFINE_GLOBAL_11
#define UPDATE_GLOBAL_11
#define REDUCE_GLOBAL_11
#endif

#if DEF_CHAIN_COUNT > 12
#define DEFINE_CHAIN_12 DEFINE_CHAIN(12)
#define RESET_CHAIN_12 RESET_CHAIN(12)
#define ACCESS_CHAIN_12 ACCESS_CHAIN(12)
#define DEFINE_GLOBAL_12 DEFINE_GLOBAL(12)
#define UPDATE_GLOBAL_12 UPDATE_GLOBAL(12)
#define REDUCE_GLOBAL_12 REDUCE_GLOBAL(12)
#else
#define DEFINE_CHAIN_12
#define RESET_CHAIN_12
#define ACCESS_CHAIN_12
#define DEFINE_GLOBAL_12
#define UPDATE_GLOBAL_12
#define REDUCE_GLOBAL_12
#endif

#if DEF_CHAIN_COUNT > 13
#define DEFINE_CHAIN_13 DEFINE_CHAIN(13)
#define RESET_CHAIN_13 RESET_CHAIN(13)
#define ACCESS_CHAIN_13 ACCESS_CHAIN(13)
#define DEFINE_GLOBAL_13 DEFINE_GLOBAL(13)
#define UPDATE_GLOBAL_13 UPDATE_GLOBAL(13)
#define REDUCE_GLOBAL_13 REDUCE_GLOBAL(13)
#else
#define DEFINE_CHAIN_13
#define RESET_CHAIN_13
#define ACCESS_CHAIN_13
#define DEFINE_GLOBAL_13
#define UPDATE_GLOBAL_13
#define REDUCE_GLOBAL_13
#endif

#if DEF_CHAIN_COUNT > 14
#define DEFINE_CHAIN_14 DEFINE_CHAIN(14)
#define RESET_CHAIN_14 RESET_CHAIN(14)
#define ACCESS_CHAIN_14 ACCESS_CHAIN(14)
#define DEFINE_GLOBAL_14 DEFINE_GLOBAL(14)
#define UPDATE_GLOBAL_14 UPDATE_GLOBAL(14)
#define REDUCE_GLOBAL_14 REDUCE_GLOBAL(14)
#else
#define DEFINE_CHAIN_14
#define RESET_CHAIN_14
#define ACCESS_CHAIN_14
#define DEFINE_GLOBAL_14
#define UPDATE_GLOBAL_14
#define REDUCE_GLOBAL_14
#endif

#if DEF_CHAIN_COUNT > 15
#define DEFINE_CHAIN_15 DEFINE_CHAIN(15)
#define RESET_CHAIN_15 RESET_CHAIN(15)
#define ACCESS_CHAIN_15 ACCESS_CHAIN(15)
#define DEFINE_GLOBAL_15 DEFINE_GLOBAL(15)
#define UPDATE_GLOBAL_15 UPDATE_GLOBAL(15)
#define REDUCE_GLOBAL_15 REDUCE_GLOBAL(15)
#else
#define DEFINE_CHAIN_15
#define RESET_CHAIN_15
#define ACCESS_CHAIN_15
#define DEFINE_GLOBAL_15
#define UPDATE_GLOBAL_15
#define REDUCE_GLOBAL_15
#endif


uint64_t memory_test_kernel_ptrchase_multichain(
    Node_t** node_list_list,
    uint64_t index_region,
    uint64_t repeat_count,
    uint64_t chains,
    FILE* cycle_file
){
    printf("In memory_test_kernel_ptrchase_multichain\n");
    printf("index_region : %ld\n", index_region);
    printf("repeat_count : %ld\n", repeat_count);
    printf("chains : %ld\n", chains);
    fflush(stdout);

    DEFINE_GLOBAL_0;
    DEFINE_GLOBAL_1;
    DEFINE_GLOBAL_2;
    DEFINE_GLOBAL_3;
    DEFINE_GLOBAL_4;
    DEFINE_GLOBAL_5;
    DEFINE_GLOBAL_6;
    DEFINE_GLOBAL_7;
    DEFINE_GLOBAL_8;
    DEFINE_GLOBAL_9;
    DEFINE_GLOBAL_10;
    DEFINE_GLOBAL_11;
    DEFINE_GLOBAL_12;
    DEFINE_GLOBAL_13;
    DEFINE_GLOBAL_14;
    DEFINE_GLOBAL_15;

#ifdef _OPENMP
    int thread_count = omp_get_max_threads();
#else
    int thread_count = 1;
#endif
    uint64_t cycles_threads[thread_count];
    double time_threads[thread_count];
    for(int i = 0; i < thread_count; ++i){
        cycles_threads[i] = 0;
        time_threads[i] = 0.;
    }

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        int thread_rank = omp_get_thread_num();
        int thread_size = omp_get_num_threads();
#else
        int thread_rank = 0;
        int thread_size = 1;
#endif
        // warm up
        DEFINE_CHAIN_0;
        DEFINE_CHAIN_1;
        DEFINE_CHAIN_2;
        DEFINE_CHAIN_3;
        DEFINE_CHAIN_4;
        DEFINE_CHAIN_5;
        DEFINE_CHAIN_6;
        DEFINE_CHAIN_7;
        DEFINE_CHAIN_8;
        DEFINE_CHAIN_9;
        DEFINE_CHAIN_10;
        DEFINE_CHAIN_11;
        DEFINE_CHAIN_12;
        DEFINE_CHAIN_13;
        DEFINE_CHAIN_14;
        DEFINE_CHAIN_15;
        for(uint64_t i = 0; i < index_region; ++i){
            ACCESS_CHAIN_0;
            ACCESS_CHAIN_1;
            ACCESS_CHAIN_2;
            ACCESS_CHAIN_3;
            ACCESS_CHAIN_4;
            ACCESS_CHAIN_5;
            ACCESS_CHAIN_6;
            ACCESS_CHAIN_7;
            ACCESS_CHAIN_8;
            ACCESS_CHAIN_9;
            ACCESS_CHAIN_10;
            ACCESS_CHAIN_11;
            ACCESS_CHAIN_12;
            ACCESS_CHAIN_13;
            ACCESS_CHAIN_14;
            ACCESS_CHAIN_15;
        }
        
        double time_start = dtime();
        uint64_t cycle_start = rdtsc();

        for(int i = 0; i < repeat_count; i++){
            // RESET_CHAIN_0;
            // RESET_CHAIN_1;
            // RESET_CHAIN_2;
            // RESET_CHAIN_3;
            // RESET_CHAIN_4;
            // RESET_CHAIN_5;
            // RESET_CHAIN_6;
            // RESET_CHAIN_7;
            // RESET_CHAIN_8;
            // RESET_CHAIN_9;
            // RESET_CHAIN_10;
            // RESET_CHAIN_11;
            // RESET_CHAIN_12;
            // RESET_CHAIN_13;
            // RESET_CHAIN_14;
            // RESET_CHAIN_15;
            for(uint64_t i = 0; i < index_region; ++i){
                ACCESS_CHAIN_0;
                ACCESS_CHAIN_1;
                ACCESS_CHAIN_2;
                ACCESS_CHAIN_3;
                ACCESS_CHAIN_4;
                ACCESS_CHAIN_5;
                ACCESS_CHAIN_6;
                ACCESS_CHAIN_7;
                ACCESS_CHAIN_8;
                ACCESS_CHAIN_9;
                ACCESS_CHAIN_10;
                ACCESS_CHAIN_11;
                ACCESS_CHAIN_12;
                ACCESS_CHAIN_13;
                ACCESS_CHAIN_14;
                ACCESS_CHAIN_15;
            }
        }

        uint64_t cycle_end = rdtsc();
        double time_end = dtime();

        cycles_threads[thread_rank] = cycle_end - cycle_start;
        time_threads[thread_rank] = time_end - time_start;

        UPDATE_GLOBAL_0;
        UPDATE_GLOBAL_1;
        UPDATE_GLOBAL_2;
        UPDATE_GLOBAL_3;
        UPDATE_GLOBAL_4;
        UPDATE_GLOBAL_5;
        UPDATE_GLOBAL_6;
        UPDATE_GLOBAL_7;
        UPDATE_GLOBAL_8;
        UPDATE_GLOBAL_9;
        UPDATE_GLOBAL_10;
        UPDATE_GLOBAL_11;
        UPDATE_GLOBAL_12;
        UPDATE_GLOBAL_13;
        UPDATE_GLOBAL_14;
        UPDATE_GLOBAL_15;
    }

    uint64_t total_cycles = 0;
    uint64_t max_cycles = 0;
    uint64_t min_cycles = 0xFFFFFFFFFFFFFFFF;

    double total_time = 0;
    double max_time = 0;
    double min_time = 100000000000000.;  

    for(int i = 0; i < thread_count; ++i){
        total_cycles += cycles_threads[i];
        max_cycles = max(max_cycles, cycles_threads[i]);
        min_cycles = min(min_cycles, cycles_threads[i]);

        total_time += time_threads[i];
        max_time = max(max_time, time_threads[i]);
        min_time = min(min_time, time_threads[i]);
    } 

    uint64_t avg_cycles = total_cycles / thread_count;
    double avg_time = total_time / thread_count;

    double cycles = avg_cycles * 1.0 / index_region / repeat_count / thread_count / chains;

    uint64_t access_region = index_region * sizeof(Node_t);

    double GBPS1 = 1.0 * thread_count * repeat_count * chains * index_region * sizeof(Node_t) / 1024./ 1024./ 1024./ avg_time;
    double GBPS2 = 64.0 / cycles * FREQUENCY_GHZ * 1e9 / 1024./1024./1024.;

    printf("avg_cycles : %ld\n", avg_cycles);
    printf("max_cycles : %ld\n", max_cycles);
    printf("min_cycles : %ld\n", min_cycles);
    printf("cycle diff : %ld\n", max_cycles - min_cycles);
    printf("avg_time : %.2lf s\n", avg_time);
    printf("frequent : %.2lf GHz\n", avg_cycles / 1e9 / avg_time);
    printf("cycles : %8.4lf \n", cycles);
    printf("GBPS1 : %8.4lf GB/s\n", GBPS1);
    printf("GBPS2 : %8.4lf GB/s\n", GBPS2);
    fflush(stdout);

    fprintf(cycle_file, "%ld %8.4lf %8.4lf\n", access_region, cycles, GBPS1);
    fflush(cycle_file);

    // cheat compiler
    uint64_t global_pre_ptr = 0;
    REDUCE_GLOBAL_0;
    REDUCE_GLOBAL_1;
    REDUCE_GLOBAL_2;
    REDUCE_GLOBAL_3;
    REDUCE_GLOBAL_4;
    REDUCE_GLOBAL_5;
    REDUCE_GLOBAL_6;
    REDUCE_GLOBAL_7;
    REDUCE_GLOBAL_8;
    REDUCE_GLOBAL_9;
    REDUCE_GLOBAL_10;
    REDUCE_GLOBAL_11;
    REDUCE_GLOBAL_12;
    REDUCE_GLOBAL_13;
    REDUCE_GLOBAL_14;
    REDUCE_GLOBAL_15;
    return global_pre_ptr;
}


uint64_t memory_test_kernel_seqential_without_ptrchase(
    uint64_t access_region,
    uint64_t index_region,
    uint64_t repeat_count,
    FILE* cycle_file
){
    access_region = access_region / 256 * 256;
    index_region = index_region / 4 * 4;
    printf("In memory_test_kernel_seqential_without_ptrchase\n");
    printf("access_region : %ld\n", access_region);
    printf("index_region : %ld\n", index_region);
    printf("repeat_count : %ld\n", repeat_count);
    fflush(stdout);
    assert(access_region % 256 == 0);
    uint64_t global_sum = 0;

#ifdef _OPENMP
    int thread_count = omp_get_max_threads();
#else
    int thread_count = 1;
#endif

    uint64_t cycles_threads[thread_count];
    double time_threads[thread_count];
    for(int i = 0; i < thread_count; ++i){
        cycles_threads[i] = 0;
        time_threads[i] = 0.;
    }

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        int thread_rank = omp_get_thread_num();
        int thread_size = omp_get_num_threads();
#else
        int thread_rank = 0;
        int thread_size = 1;
#endif
        
        uint64_t data_len = access_region / sizeof(uint64_t);
        uint64_t* data = aligned_alloc(64, data_len * sizeof(uint64_t));
        for(uint64_t i = 0; i < data_len; ++i){
            data[i] = i;
        }

        // warm up
        uint64_t sum0 = 0;
        uint64_t sum1 = 0;
        uint64_t sum2 = 0;
        uint64_t sum3 = 0;
        for(uint64_t i = 0; i < data_len; i += 32){
            sum0 += data[i];
            sum1 += data[i + 8];
            sum2 += data[i + 16];
            sum3 += data[i + 24];
        }
        double time_start = dtime();
        uint64_t cycle_start = rdtsc();
        for(int r = 0; r < repeat_count; r++){
            for(uint64_t i = 0; i < data_len; i += 32){
                sum0 += data[i];
                sum1 += data[i + 8];
                sum2 += data[i + 16];
                sum3 += data[i + 24];
            }
        }
        uint64_t cycle_end = rdtsc();
        double time_end = dtime();

        free(data);

        cycles_threads[thread_rank] = cycle_end - cycle_start;
        time_threads[thread_rank] = time_end - time_start;

        uint64_t local_sum = sum0 + sum1 + sum2 + sum3;

#ifdef _OPENMP
#pragma omp atomic
#endif
        global_sum += local_sum;
    }

    uint64_t total_cycles = 0;
    uint64_t max_cycles = 0;
    uint64_t min_cycles = 0xFFFFFFFFFFFFFFFF;

    double total_time = 0;
    double max_time = 0;
    double min_time = 100000000000000.;

    for(int i = 0; i < thread_count; ++i){
        total_cycles += cycles_threads[i];
        max_cycles = max(max_cycles, cycles_threads[i]);
        min_cycles = min(min_cycles, cycles_threads[i]);

        total_time += time_threads[i];
        max_time = max(max_time, time_threads[i]);
        min_time = min(min_time, time_threads[i]);
    }
    uint64_t avg_cycles = total_cycles / thread_count;
    double avg_time = total_time / thread_count;

    double cycles = avg_cycles * 1.0 / index_region / repeat_count / thread_count;
    double GBPS1 = 1.0 * thread_count * repeat_count * index_region * CACHE_LINE_SIZE / 1024./1024./1024. / avg_time;
    double GBPS2 = 1.0 * CACHE_LINE_SIZE / cycles * FREQUENCY_GHZ * 1e9 / 1024./1024./1024.;

    printf("avg_cycles : %ld\n", avg_cycles);
    printf("max_cycles : %ld\n", max_cycles);
    printf("min_cycles : %ld\n", min_cycles);
    printf("cycle diff : %ld\n", max_cycles - min_cycles);
    printf("avg_time : %.2lf s\n", avg_time);
    printf("frequent : %.2lf GHz\n", avg_cycles / 1e9 / avg_time);
    printf("GBPS1 : %8.4lf GB/s\n", GBPS1);
    printf("GBPS2 : %8.4lf GB/s\n", GBPS2);
    fflush(stdout);

    fprintf(cycle_file, "%ld %8.4lf %8.4lf\n", access_region, cycles, GBPS1);
    fflush(cycle_file);

    return global_sum;
}


uint64_t memory_test_kernel_random_without_ptrchase(
    uint64_t** ptrs,
    uint64_t access_region,
    uint64_t index_region,
    uint64_t repeat_count,
    FILE* cycle_file
){
    access_region = access_region / 256 * 256;
    index_region = index_region / 4 * 4;
    assert(access_region % 256 == 0);
    assert(index_region % 4 == 0);

    printf("In memory_test_kernel_random_without_ptrchase\n");
    printf("access_region : %ld\n", access_region);
    printf("index_region : %ld\n", index_region);
    printf("repeat_count : %ld\n", repeat_count);
    fflush(stdout);

    uint64_t global_sum = 0;
#ifdef _OPENMP
    int thread_count = omp_get_max_threads();
#else
    int thread_count = 1;
#endif

    uint64_t cycles_threads[thread_count];
    double time_threads[thread_count];
    for(int i = 0; i < thread_count; ++i){
        cycles_threads[i] = 0;
        time_threads[i] = 0.;
    }

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {

#ifdef _OPENMP
        int thread_rank = omp_get_thread_num();
        int thread_size = omp_get_num_threads();
#else
        int thread_rank = 0;
        int thread_size = 1;
#endif
        uint64_t data_len = access_region / sizeof(uint64_t);
        uint64_t* data = aligned_alloc(64 ,sizeof(uint64_t) * data_len);
        for(uint64_t i = 0; i < data_len; ++i){
            data[i] = i;
        }
        // warm up
        uint64_t* ptr0 = ptrs[0];
        uint64_t sum0 = 0;
        uint64_t sum1 = 0;
        uint64_t sum2 = 0;
        uint64_t sum3 = 0;
        for(uint64_t i = 0; i < index_region; i += 4){
            sum0 += data[ptr0[i + 0] * 8];
            sum1 += data[ptr0[i + 1] * 8];
            sum2 += data[ptr0[i + 2] * 8];
            sum3 += data[ptr0[i + 3] * 8];
        }
        double time_start = dtime();
        uint64_t cycle_start = rdtsc();

        for(int r = 0; r < repeat_count; r++){
            for(uint64_t i = 0; i < index_region; i += 4){
                sum0 += data[ptr0[i + 0] * 8];
                sum1 += data[ptr0[i + 1] * 8];
                sum2 += data[ptr0[i + 2] * 8];
                sum3 += data[ptr0[i + 3] * 8];
            }
        }

        uint64_t cycle_end = rdtsc();
        double time_end = dtime();

        free(data);
        cycles_threads[thread_rank] = cycle_end - cycle_start;     
        time_threads[thread_rank] = time_end - time_start;

        uint64_t local_sum = sum0 + sum1 + sum2 + sum3;
    
#ifdef _OPENMP
#pragma omp atomic
#endif
        global_sum += local_sum;
    }

    uint64_t total_cycles = 0;
    uint64_t max_cycles = 0;
    uint64_t min_cycles = 0xFFFFFFFFFFFFFFFF;

    double total_time = 0;
    double max_time = 0;
    double min_time = 100000000000000.;

    for(int i = 0; i < thread_count; ++i){
        total_cycles += cycles_threads[i];
        max_cycles = max(max_cycles, cycles_threads[i]);
        min_cycles = min(min_cycles, cycles_threads[i]);

        total_time += time_threads[i];
        max_time = max(max_time, time_threads[i]);
        min_time = min(min_time, time_threads[i]);
    }
    uint64_t avg_cycles = total_cycles / thread_count;
    double avg_time = total_time / thread_count;

    double cycles = avg_cycles * 1.0 / index_region / repeat_count / thread_count;
    double GBPS1 = 1.0 * thread_count * repeat_count * index_region * CACHE_LINE_SIZE / 1024./1024./1024. / avg_time;
    double GBPS2 = 1.0 * CACHE_LINE_SIZE / cycles * FREQUENCY_GHZ * 1e9 / 1024./1024./1024.;

    printf("avg_cycles : %ld\n", avg_cycles);
    printf("max_cycles : %ld\n", max_cycles);
    printf("min_cycles : %ld\n", min_cycles);
    printf("cycle diff : %ld\n", max_cycles - min_cycles);
    printf("avg_time : %.2lf s\n", avg_time);
    printf("frequent : %.2lf GHz\n", avg_cycles / 1e9 / avg_time);
    printf("GBPS1 : %8.4lf GB/s\n", GBPS1);
    printf("GBPS2 : %8.4lf GB/s\n", GBPS2);
    fflush(stdout);

    fprintf(cycle_file, "%ld %8.4lf %8.4lf\n", access_region, cycles, GBPS1);
    fflush(cycle_file);

    return global_sum;
}
