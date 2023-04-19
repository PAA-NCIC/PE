#ifndef _CPUBM_X86_HPP
#define _CPUBM_X86_HPP

#include <string>
#include <vector>

void reg_new_fp_bench(std::string isa,
    std::string type,
    std::string dim,
    int64_t num_loops,
    int64_t flops_per_loop,
    void (*bench)(int64_t));

void reg_new_mem_bench(std::string isa,
    std::string type,
    std::string dim,
    int64_t num_loops,
    int64_t data_volume_per_loop,
    void (*bench)(int64_t, void*, void*, void*));
    
void cpubm_do_bench(std::vector<int> &set_of_threads);

#endif

