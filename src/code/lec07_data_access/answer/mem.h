#pragma once

#include "util.h"

typedef void* pointer;

#define CACHE_LINE_SIZE 64

#define FREQUENCY_GHZ 2.9

typedef struct {
    pointer next;
    char padding[CACHE_LINE_SIZE - 1 * sizeof(pointer)];
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

static void release_access_list(uint64_t* ptr){
    free(ptr);
}

static void gen_node_list(Node_t** node_list_p, uint64_t* ptr, uint64_t index_region){
    *node_list_p = aligned_alloc(64, sizeof(Node_t) * index_region);
    Node_t* node_list = *node_list_p;
    for(uint64_t i = 0; i < index_region; ++i){
        node_list[i].next = &node_list[ptr[i]];
    }
}

static void release_node_list(Node_t* node_list){
    free(node_list);
}

