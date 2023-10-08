#pragma once

#include "util.h"

#ifndef DEF_CHAIN_COUNT
#define DEF_CHAIN_COUNT 1
#endif

#if !defined(DEF_GEN_RANDOM_LIST) && !defined(DEF_GEN_SEQUENTIAL_LIST)
#define DEF_GEN_RANDOM_LIST
#endif

typedef void* pointer;

#ifdef DEF_PREFETCH

typedef struct {
    pointer next;
    pointer prefetch;
} Node_t;

#endif

static void gen_random_list(uint64_t** ptr_p, uint64_t index_region){
    printf("index_region : %ld\n", index_region);
    fflush(stdout);
    *ptr_p = malloc(sizeof(uint64_t) * index_region);
    uint64_t* ptr = *ptr_p;
    bool* access = malloc(sizeof(bool) * index_region);
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
    *ptr_p = malloc(sizeof(uint64_t) * index_region);
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
    *ptrs_p = malloc(sizeof(uint64_t*) * chains);
    uint64_t** ptrs = *ptrs_p;
    for(uint64_t c = 0; c < chains; ++c){
        printf("number of chain  : %ld\n", chains);
        printf("chain number : %ld\n", c);
        fflush(stdout);
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

#ifdef DEF_PREFETCH

static void gen_node_list(Node_t** node_list_p, uint64_t* ptr, uint64_t index_region,uint64_t prefetch_count){
    *node_list_p = malloc(sizeof(Node_t) * index_region);
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
    *node_list_list_p = malloc(sizeof(Node_t*) * chains);
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

#endif

uint64_t memory_test_kernel_ptrchase_multichain(
    uint64_t** ptrs,
    uint64_t index_region,
    uint64_t access_count,
    uint64_t repeat_count,
    uint64_t chains,
    FILE* latency_file
){
#if DEF_CHAIN_COUNT > 0
    uint64_t global_pre_ptr_0 = 0;
#endif
#if DEF_CHAIN_COUNT > 1
    uint64_t global_pre_ptr_1 = 0;
#endif
#if DEF_CHAIN_COUNT > 2
    uint64_t global_pre_ptr_2 = 0;
#endif
#if DEF_CHAIN_COUNT > 3
    uint64_t global_pre_ptr_3 = 0;
#endif
#if DEF_CHAIN_COUNT > 4
    uint64_t global_pre_ptr_4 = 0;
#endif
#if DEF_CHAIN_COUNT > 5
    uint64_t global_pre_ptr_5 = 0;
#endif
#if DEF_CHAIN_COUNT > 6
    uint64_t global_pre_ptr_6 = 0;
#endif
#if DEF_CHAIN_COUNT > 7
    uint64_t global_pre_ptr_7 = 0;
#endif
#if DEF_CHAIN_COUNT > 8
    uint64_t global_pre_ptr_8 = 0;
#endif
#if DEF_CHAIN_COUNT > 9
    uint64_t global_pre_ptr_9 = 0;
#endif
#if DEF_CHAIN_COUNT > 10
    uint64_t global_pre_ptr_10 = 0;
#endif
#if DEF_CHAIN_COUNT > 11
    uint64_t global_pre_ptr_11 = 0;
#endif
#if DEF_CHAIN_COUNT > 12
    uint64_t global_pre_ptr_12 = 0;
#endif
#if DEF_CHAIN_COUNT > 13
    uint64_t global_pre_ptr_13 = 0;
#endif
#if DEF_CHAIN_COUNT > 14
    uint64_t global_pre_ptr_14 = 0;
#endif
#if DEF_CHAIN_COUNT > 15
    uint64_t global_pre_ptr_15 = 0;
#endif

#ifdef _OPENMP
    int thread_count = omp_get_max_threads();
#else
    int thread_count = 1;
#endif
    uint64_t cycles_threads[thread_count];
    for(int i = 0; i < thread_count; ++i){
        cycles_threads[i] = 0;
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
#if DEF_CHAIN_COUNT > 0
        uint64_t pre_ptr_0 = 0;
        uint64_t* ptr0 = ptrs[0];
#endif
#if DEF_CHAIN_COUNT > 1
        uint64_t pre_ptr_1 = 0;
        uint64_t* ptr1 = ptrs[1];
#endif
#if DEF_CHAIN_COUNT > 2
        uint64_t pre_ptr_2 = 0;
        uint64_t* ptr2 = ptrs[2];
#endif
#if DEF_CHAIN_COUNT > 3
        uint64_t pre_ptr_3 = 0;
        uint64_t* ptr3 = ptrs[3];
#endif
#if DEF_CHAIN_COUNT > 4
        uint64_t pre_ptr_4 = 0;
        uint64_t* ptr4 = ptrs[4];
#endif
#if DEF_CHAIN_COUNT > 5
        uint64_t pre_ptr_5 = 0;
        uint64_t* ptr5 = ptrs[5];
#endif
#if DEF_CHAIN_COUNT > 6
        uint64_t pre_ptr_6 = 0;
        uint64_t* ptr6 = ptrs[6];
#endif
#if DEF_CHAIN_COUNT > 7
        uint64_t pre_ptr_7 = 0;
        uint64_t* ptr7 = ptrs[7];
#endif
#if DEF_CHAIN_COUNT > 8
        uint64_t pre_ptr_8 = 0;
        uint64_t* ptr8 = ptrs[8];
#endif
#if DEF_CHAIN_COUNT > 9
        uint64_t pre_ptr_9 = 0;
        uint64_t* ptr9 = ptrs[9];
#endif
#if DEF_CHAIN_COUNT > 10
        uint64_t pre_ptr_10 = 0;
        uint64_t* ptr10 = ptrs[10];
#endif
#if DEF_CHAIN_COUNT > 11
        uint64_t pre_ptr_11 = 0;
        uint64_t* ptr11 = ptrs[11];
#endif
#if DEF_CHAIN_COUNT > 12
        uint64_t pre_ptr_12 = 0;
        uint64_t* ptr12 = ptrs[12];
#endif
#if DEF_CHAIN_COUNT > 13
        uint64_t pre_ptr_13 = 0;
        uint64_t* ptr13 = ptrs[13];
#endif
#if DEF_CHAIN_COUNT > 14
        uint64_t pre_ptr_14 = 0;
        uint64_t* ptr14 = ptrs[14];
#endif
#if DEF_CHAIN_COUNT > 15
        uint64_t pre_ptr_15 = 0;
        uint64_t* ptr15 = ptrs[15];
#endif
        for(uint64_t i = 0; i < access_count; ++i){
#if DEF_CHAIN_COUNT > 0
            pre_ptr_0 = ptr0[pre_ptr_0];
#endif
#if DEF_CHAIN_COUNT > 1
            pre_ptr_1 = ptr1[pre_ptr_1];
#endif
#if DEF_CHAIN_COUNT > 2
            pre_ptr_2 = ptr2[pre_ptr_2];
#endif
#if DEF_CHAIN_COUNT > 3
            pre_ptr_3 = ptr3[pre_ptr_3];
#endif
#if DEF_CHAIN_COUNT > 4
            pre_ptr_4 = ptr4[pre_ptr_4];
#endif
#if DEF_CHAIN_COUNT > 5
            pre_ptr_5 = ptr5[pre_ptr_5];
#endif
#if DEF_CHAIN_COUNT > 6
            pre_ptr_6 = ptr6[pre_ptr_6];
#endif
#if DEF_CHAIN_COUNT > 7
            pre_ptr_7 = ptr7[pre_ptr_7];
#endif
#if DEF_CHAIN_COUNT > 8
            pre_ptr_8 = ptr8[pre_ptr_8];
#endif
#if DEF_CHAIN_COUNT > 9
            pre_ptr_9 = ptr9[pre_ptr_9];
#endif
#if DEF_CHAIN_COUNT > 10
            pre_ptr_10 = ptr10[pre_ptr_10];
#endif
#if DEF_CHAIN_COUNT > 11
            pre_ptr_11 = ptr11[pre_ptr_11];
#endif
#if DEF_CHAIN_COUNT > 12
            pre_ptr_12 = ptr12[pre_ptr_12];
#endif
#if DEF_CHAIN_COUNT > 13
            pre_ptr_13 = ptr13[pre_ptr_13];
#endif
#if DEF_CHAIN_COUNT > 14
            pre_ptr_14 = ptr14[pre_ptr_14];
#endif
#if DEF_CHAIN_COUNT > 15
            pre_ptr_15 = ptr15[pre_ptr_15];
#endif
        }
        uint64_t cycle_start = rdtsc();

        for(int i = 0; i < repeat_count; i++){
#if DEF_CHAIN_COUNT > 0
            pre_ptr_0 = 0;
#endif
#if DEF_CHAIN_COUNT > 1
            pre_ptr_1 = 0;
#endif
#if DEF_CHAIN_COUNT > 2
            pre_ptr_2 = 0;
#endif
#if DEF_CHAIN_COUNT > 3
            pre_ptr_3 = 0;
#endif
#if DEF_CHAIN_COUNT > 4
            pre_ptr_4 = 0;
#endif
#if DEF_CHAIN_COUNT > 5
            pre_ptr_5 = 0;
#endif
#if DEF_CHAIN_COUNT > 6
            pre_ptr_6 = 0;
#endif
#if DEF_CHAIN_COUNT > 7
            pre_ptr_7 = 0;
#endif
#if DEF_CHAIN_COUNT > 8
            pre_ptr_8 = 0;
#endif
#if DEF_CHAIN_COUNT > 9
            pre_ptr_9 = 0;
#endif
#if DEF_CHAIN_COUNT > 10
            pre_ptr_10 = 0;
#endif
#if DEF_CHAIN_COUNT > 11
            pre_ptr_11 = 0;
#endif
#if DEF_CHAIN_COUNT > 12
            pre_ptr_12 = 0;
#endif
#if DEF_CHAIN_COUNT > 13
            pre_ptr_13 = 0;
#endif
#if DEF_CHAIN_COUNT > 14
            pre_ptr_14 = 0;
#endif
#if DEF_CHAIN_COUNT > 15
            pre_ptr_15 = 0;
#endif
            for(uint64_t i = 0; i < access_count; ++i){
#if DEF_CHAIN_COUNT > 0
                pre_ptr_0 = ptr0[pre_ptr_0];
#endif
#if DEF_CHAIN_COUNT > 1
                pre_ptr_1 = ptr1[pre_ptr_1];
#endif
#if DEF_CHAIN_COUNT > 2
                pre_ptr_2 = ptr2[pre_ptr_2];
#endif
#if DEF_CHAIN_COUNT > 3
                pre_ptr_3 = ptr3[pre_ptr_3];
#endif
#if DEF_CHAIN_COUNT > 4
                pre_ptr_4 = ptr4[pre_ptr_4];
#endif
#if DEF_CHAIN_COUNT > 5
                pre_ptr_5 = ptr5[pre_ptr_5];
#endif
#if DEF_CHAIN_COUNT > 6
                pre_ptr_6 = ptr6[pre_ptr_6];
#endif
#if DEF_CHAIN_COUNT > 7
                pre_ptr_7 = ptr7[pre_ptr_7];
#endif
#if DEF_CHAIN_COUNT > 8
                pre_ptr_8 = ptr8[pre_ptr_8];
#endif
#if DEF_CHAIN_COUNT > 9
                pre_ptr_9 = ptr9[pre_ptr_9];
#endif
#if DEF_CHAIN_COUNT > 10
                pre_ptr_10 = ptr10[pre_ptr_10];
#endif
#if DEF_CHAIN_COUNT > 11
                pre_ptr_11 = ptr11[pre_ptr_11];
#endif
#if DEF_CHAIN_COUNT > 12
                pre_ptr_12 = ptr12[pre_ptr_12];
#endif
#if DEF_CHAIN_COUNT > 13
                pre_ptr_13 = ptr13[pre_ptr_13];
#endif
#if DEF_CHAIN_COUNT > 14
                pre_ptr_14 = ptr14[pre_ptr_14];
#endif
#if DEF_CHAIN_COUNT > 15
                pre_ptr_15 = ptr15[pre_ptr_15];
#endif
            }
        }
        uint64_t cycle_end = rdtsc();

        cycles_threads[thread_rank] = cycle_end - cycle_start;

#if DEF_CHAIN_COUNT > 0
        global_pre_ptr_0 += pre_ptr_0;
#endif
#if DEF_CHAIN_COUNT > 1
        global_pre_ptr_1 += pre_ptr_1;
#endif
#if DEF_CHAIN_COUNT > 2
        global_pre_ptr_2 += pre_ptr_2;
#endif
#if DEF_CHAIN_COUNT > 3
        global_pre_ptr_3 += pre_ptr_3;
#endif
#if DEF_CHAIN_COUNT > 4
        global_pre_ptr_4 += pre_ptr_4;
#endif
#if DEF_CHAIN_COUNT > 5
        global_pre_ptr_5 += pre_ptr_5;
#endif
#if DEF_CHAIN_COUNT > 6
        global_pre_ptr_6 += pre_ptr_6;
#endif
#if DEF_CHAIN_COUNT > 7
        global_pre_ptr_7 += pre_ptr_7;
#endif
#if DEF_CHAIN_COUNT > 8
        global_pre_ptr_8 += pre_ptr_8;
#endif
#if DEF_CHAIN_COUNT > 9
        global_pre_ptr_9 += pre_ptr_9;
#endif
#if DEF_CHAIN_COUNT > 10
        global_pre_ptr_10 += pre_ptr_10;
#endif
#if DEF_CHAIN_COUNT > 11
        global_pre_ptr_11 += pre_ptr_11;
#endif
#if DEF_CHAIN_COUNT > 12
        global_pre_ptr_12 += pre_ptr_12;
#endif
#if DEF_CHAIN_COUNT > 13
        global_pre_ptr_13 += pre_ptr_13;
#endif
#if DEF_CHAIN_COUNT > 14
        global_pre_ptr_14 += pre_ptr_14;
#endif
#if DEF_CHAIN_COUNT > 15
        global_pre_ptr_15 += pre_ptr_15;
#endif
    }
    
    uint64_t total_cycles = 0;
    for(int i = 0; i < thread_count; ++i){
        total_cycles += cycles_threads[i];
    }
    uint64_t avg_cycles = total_cycles / thread_count;

    // cycles per load
    double cycles = avg_cycles * 1. / access_count / thread_count / repeat_count / chains;

    fprintf(latency_file, "%ld %8.4lf\n", sizeof(uint64_t) * index_region, cycles);
    fflush(latency_file);

    // cheat compiler
    uint64_t global_pre_ptr = 0;
#if DEF_CHAIN_COUNT > 0 
    global_pre_ptr += global_pre_ptr_0;
#endif
#if DEF_CHAIN_COUNT > 1 
    global_pre_ptr += global_pre_ptr_1;
#endif
#if DEF_CHAIN_COUNT > 2 
    global_pre_ptr += global_pre_ptr_2;
#endif
#if DEF_CHAIN_COUNT > 3 
    global_pre_ptr += global_pre_ptr_3;
#endif
#if DEF_CHAIN_COUNT > 4 
    global_pre_ptr += global_pre_ptr_4;
#endif
#if DEF_CHAIN_COUNT > 5 
    global_pre_ptr += global_pre_ptr_5;
#endif
#if DEF_CHAIN_COUNT > 6 
    global_pre_ptr += global_pre_ptr_6;
#endif
#if DEF_CHAIN_COUNT > 7 
    global_pre_ptr += global_pre_ptr_7;
#endif
#if DEF_CHAIN_COUNT > 8 
    global_pre_ptr += global_pre_ptr_8;
#endif
#if DEF_CHAIN_COUNT > 9 
    global_pre_ptr += global_pre_ptr_9;
#endif
#if DEF_CHAIN_COUNT > 10 
    global_pre_ptr += global_pre_ptr_10;
#endif
#if DEF_CHAIN_COUNT > 11 
    global_pre_ptr += global_pre_ptr_11;
#endif
#if DEF_CHAIN_COUNT > 12 
    global_pre_ptr += global_pre_ptr_12;
#endif
#if DEF_CHAIN_COUNT > 13 
    global_pre_ptr += global_pre_ptr_13;
#endif
#if DEF_CHAIN_COUNT > 14 
    global_pre_ptr += global_pre_ptr_14;
#endif
#if DEF_CHAIN_COUNT > 15 
    global_pre_ptr += global_pre_ptr_15;
#endif
    return global_pre_ptr;
}

#ifdef DEF_PREFETCH 
uint64_t memory_test_kernel_ptrchase_prefetch_multichain(
    Node_t** node_list_list,
    uint64_t index_region,
    uint64_t access_count,
    uint64_t repeat_count,
    FILE* cycle_file
){
    // warm up
    Node_t* node_list_0 = node_list_list[0];
    Node_t* pre_ptr_0 = &node_list_0[0];
    for(uint64_t i = 0; i < access_count; ++i){
        pre_ptr_0 = (Node_t*)((pre_ptr_0->next));
        __builtin_prefetch((Node_t*)(pre_ptr_0->prefetch));
    }
    uint64_t cycle_start = rdtsc();

    for(int i = 0; i < repeat_count; i++){
        pre_ptr_0 = &node_list_0[0];
        for(uint64_t i = 0; i < access_count; ++i){
            pre_ptr_0 = (Node_t*)(pre_ptr_0->next);
            __builtin_prefetch((Node_t*)(pre_ptr_0->prefetch));
        }
    }
    uint64_t cycle_end = rdtsc();

    // cycles per load
    double cycles = (cycle_end - cycle_start) * 1.0 / access_count / repeat_count;

    uint64_t access_region = index_region * sizeof(Node_t);

    fprintf(cycle_file, "%ld %8.4lf\n", access_region, cycles);
    fflush(cycle_file);

    // cheat compiler
    uint64_t pre_ptr = 0;
    pre_ptr += (uint64_t)pre_ptr_0;
    return pre_ptr;
}

#endif

uint64_t memory_test_kernel_seqential_without_ptrchase(
    uint64_t index_region,
    uint64_t repeat_count,
    FILE* cycle_file
){
    index_region = index_region / 4 * 4;

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

        uint64_t* data = malloc(sizeof(uint64_t) * index_region);
        for(uint64_t i = 0; i < index_region; ++i){
            data[i] = i;
        }
        // warm up
        uint64_t sum0 = 0;
        uint64_t sum1 = 0;
        uint64_t sum2 = 0;
        uint64_t sum3 = 0;
        for(uint64_t i = 0; i < index_region; i += 4){
            sum0 += data[i + 0];
            sum1 += data[i + 1];
            sum2 += data[i + 2];
            sum3 += data[i + 3];
        }

        double time_start = dtime();
        uint64_t cycle_start = rdtsc();
        for(int r = 0; r < repeat_count; r++){
            sum0 = 0;
            sum1 = 0;
            sum2 = 0;
            sum3 = 0;
            for(uint64_t i = 0; i < index_region; i += 4){
                sum0 += data[i + 0];
                sum1 += data[i + 1];
                sum2 += data[i + 2];
                sum3 += data[i + 3];
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

    printf("avg_cycles : %ld\n", avg_cycles);
    printf("max_cycles : %ld\n", max_cycles);
    printf("min_cycles : %ld\n", min_cycles);
    printf("cycle diff : %ld\n", max_cycles - min_cycles);
    printf("avg_time : %.2lf s\n", avg_time);
    printf("frequent : %.2lf GHz\n", avg_cycles / 1e9 / avg_time);
    fflush(stdout);

    // cycles per load
    double cycles = avg_cycles * 1.0 / index_region / repeat_count / thread_count;
    
    fprintf(cycle_file, "%ld %8.4lf\n", sizeof(uint64_t) * index_region, cycles);
    fflush(cycle_file);

    return global_sum;
}


uint64_t memory_test_kernel_random_without_ptrchase(
    uint64_t** ptrs,
    uint64_t index_region,
    uint64_t repeat_count,
    FILE* cycle_file
){
    index_region = index_region / 4 * 4;

    assert(index_region % 4 == 0);

    uint64_t global_sum = 0;
#ifdef _OPENMP
    int thread_count = omp_get_max_threads();
#else
    int thread_count = 1;
#endif

    uint64_t cycles_threads[thread_count];
    for(int i = 0; i < thread_count; ++i){
        cycles_threads[i] = 0;
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
        uint64_t* data = malloc(sizeof(uint64_t) * index_region);
        for(uint64_t i = 0; i < index_region; ++i){
            data[i] = i;
        }
        // warm up
        uint64_t* ptr0 = ptrs[0];
        uint64_t sum0 = 0;
        uint64_t sum1 = 0;
        uint64_t sum2 = 0;
        uint64_t sum3 = 0;
        for(uint64_t i = 0; i < index_region; i += 4){
            sum0 += data[ptr0[i + 0]];
            sum1 += data[ptr0[i + 1]];
            sum2 += data[ptr0[i + 2]];
            sum3 += data[ptr0[i + 3]];
        }
        double time_start = dtime();
        uint64_t cycle_start = rdtsc();

        for(int r = 0; r < repeat_count; r++){
            sum0 = 0;
            sum1 = 0;
            sum2 = 0;
            sum3 = 0;
            for(uint64_t i = 0; i < index_region; i += 4){
                sum0 += data[ptr0[i + 0]];
                sum1 += data[ptr0[i + 1]];
                sum2 += data[ptr0[i + 2]];
                sum3 += data[ptr0[i + 3]];
            }
        }

        uint64_t cycle_end = rdtsc();
        double time_end = dtime();
        double time = time_end - time_start;

        free(data);
        cycles_threads[thread_rank] = cycle_end - cycle_start;     
        uint64_t local_sum = sum0 + sum1 + sum2 + sum3;
    
#ifdef _OPENMP
#pragma omp atomic
#endif
        global_sum += local_sum;
    }
    
    uint64_t total_cycles = 0;
    for(int i = 0; i < thread_count; ++i){
        total_cycles += cycles_threads[i];
    }
    uint64_t avg_cycles = total_cycles / thread_count;

    // cycles per load
    double cycles = avg_cycles * 1.0 / index_region / repeat_count / thread_count;

    fprintf(cycle_file, "%ld %8.4lf\n", sizeof(uint64_t) * 2 * index_region, cycles);
    fflush(cycle_file);

    return global_sum;
}
