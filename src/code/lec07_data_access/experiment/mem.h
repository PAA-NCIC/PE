#pragma once

#include "util.h"

typedef void* pointer;

typedef struct {
    pointer next;
    char padding[CACHE_LINE_SIZE - 1 * sizeof(pointer)];
} Node_t;

static void gen_random_list(uint64_t** ptr_p, uint64_t index_region){
    printf("index_region : %ld\n", index_region);
    fflush(stdout);
    *ptr_p = (uint64_t*)aligned_alloc(64, sizeof(uint64_t) * index_region);
    uint64_t* ptr = *ptr_p;

    // 这是一个大小为index_region的环，但是不是随机的，每个指针指向它的下一个指针。
    for(uint64_t i = 0; i < index_region; ++i){
        ptr[i] = (i + 1) % index_region;
    }
    
    // 生成一个随机数列表，构成一个大小为其长度的环。即每个指针指向一个随机指针。
    ...

    fflush(stdout);
}

static void release_access_list(uint64_t* ptr){
    free(ptr);
}

static void gen_node_list(Node_t** node_list_p, uint64_t* ptr, uint64_t index_region){
    *node_list_p = (Node_t*)aligned_alloc(64, sizeof(Node_t) * index_region);
    Node_t* node_list = *node_list_p;

    // 使用ptr，生成 node list，
    ...
}

static void release_node_list(Node_t* node_list){
    free(node_list);
}

