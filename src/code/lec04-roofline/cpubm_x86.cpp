#include "table.hpp"
#include "smtl.hpp"
#include "cpubm_x86.hpp"
#include "benchtypes.hpp"

#include <cstdint>
#include <ctime>
#include <vector>
#include <sstream>
#include <iomanip>
#include <string.h>
using namespace std;




static double get_time(struct timespec *start,
	struct timespec *end)
{
	return end->tv_sec - start->tv_sec +
		(end->tv_nsec - start->tv_nsec) * 1e-9;
}

static vector<cpu_fp_x86> fp_bench_list;
static vector<cpu_mem_x86> mem_bench_list;

void reg_new_fp_bench(std::string isa,
    std::string type,
    std::string dim,
    int64_t num_loops,
    int64_t flops_per_loop,
    void (*bench)(int64_t))
{
    //peak fp 
    cpu_fp_x86 new_one;
    new_one.isa = isa; 
    new_one.type = type;
    new_one.dim = dim;
    new_one.num_loops = num_loops;
    new_one.flops_per_loop = flops_per_loop;
    new_one.bench = bench;
    fp_bench_list.push_back(new_one);
}

static void thread_bench_fp(void *params)
{
    cpu_fp_x86 *bm = (cpu_fp_x86 *)params;
    bm->bench(bm->num_loops);
}

void reg_new_mem_bench(std::string isa,
    std::string type,
    std::string dim,
    int32_t num_loops,
    int32_t dv_per_loop,
    void (*bench)(int64_t, void *))
{
    //peak mem
    cpu_mem_x86 mem_bench_item;
    mem_bench_item.isa = isa;
    mem_bench_item.type = type;
    mem_bench_item.dim = dim;
    mem_bench_item.num_loops = num_loops;
    mem_bench_item.dv_per_loop = dv_per_loop;
    mem_bench_item.bench = bench;
    mem_bench_list.push_back(mem_bench_item);
}

static void thread_bench_mem(void *params)
{
    cpu_mem_x86 *bm = (cpu_mem_x86 *)params;
    //printf("%u\n", bm->num_loops);
    for(uint32_t i = 0; i < bm->num_loops; i++) {
        bm->bench(bm->dv_per_loop, bm->src1);
    }
}

static void pe_execute_bench(smtl_handle sh,
    BENCH_TYPE bench_type,
    void* params,
    Table &table)
{
    struct timespec start, end;
    double time_used, perf;
    int i;
    int num_threads = smtl_num_threads(sh);
    switch (bench_type) {
        case PEAK_FP: {
            cpu_fp_x86 *item = (cpu_fp_x86 *)params;
            // warm up
            for (i = 0; i < num_threads; i++)
            {
                smtl_add_task(sh, thread_bench_fp, params);
            }
            smtl_begin_tasks(sh);
            smtl_wait_tasks_finished(sh);

            //bench
            for (i = 0; i < num_threads; i++)
            {
                smtl_add_task(sh, thread_bench_fp, params);
            }
            clock_gettime(CLOCK_MONOTONIC_RAW, &start);
            smtl_begin_tasks(sh);
            smtl_wait_tasks_finished(sh);
            clock_gettime(CLOCK_MONOTONIC_RAW, &end);

            time_used = get_time(&start, &end);
            //printf("time_used:%lf\n", time_used);
            perf = item->num_loops * item->flops_per_loop * num_threads /
                time_used * 1e-9;
            
            stringstream ss;
            ss << std::setprecision(5) << perf << " " << item->dim;
            vector<string> cont;
            cont.resize(3);
            cont[0] = item->isa;
            cont[1] = item->type;
            cont[2] = ss.str();
            table.addOneItem(cont);
            break;
        }
        case PEAK_MEM: {
            void *src;
            int64_t total_size = (int64_t)(((cpu_mem_x86 *)params)->dv_per_loop) * num_threads;
            //printf("%lu\n", total_size);
            src = aligned_alloc(512, total_size);
            memset(src, 0, total_size);
            cpu_mem_x86 *item = (cpu_mem_x86 *)malloc(sizeof(cpu_mem_x86) * num_threads);
            // warm up
            for (i = 0; i < num_threads; i++)
            {
                item[i].num_loops = ((cpu_mem_x86*)params)->num_loops;
                item[i].dv_per_loop = ((cpu_mem_x86*)params)->dv_per_loop;
                item[i].bench = ((cpu_mem_x86*)params)->bench;
                item[i].src1 = (void *)((char *)src + i * item[i].dv_per_loop);
                //printf("%u, %p\n",  item[i].dv_per_loop,  item[i].src1);
                smtl_add_task(sh, thread_bench_mem, (void*)&item[i]);
            }
            smtl_begin_tasks(sh);
            smtl_wait_tasks_finished(sh);

            //bench
            for (i = 0; i < num_threads; i++)
            {
                smtl_add_task(sh, thread_bench_mem, (void*)&item[i]);
            }
            clock_gettime(CLOCK_MONOTONIC_RAW, &start);
            smtl_begin_tasks(sh);
            smtl_wait_tasks_finished(sh);
            clock_gettime(CLOCK_MONOTONIC_RAW, &end);
            time_used = get_time(&start, &end);
            //printf("time_used:%lf\n", time_used);
            perf = total_size * item[0].num_loops / time_used * 1e-9;
            
            free(item);

            stringstream ss;
            ss << std::setprecision(5) << perf << " " << ((cpu_mem_x86 *)params)->dim;
            vector<string> cont;
            cont.resize(3);
            cont[0] = ((cpu_mem_x86 *)params)->isa;
            cont[1] = ((cpu_mem_x86 *)params)->type;
            cont[2] = ss.str();
            table.addOneItem(cont);
            break;
        }
        break;
        case CUSTOM:
        break;
    }

	
}

void pe_bench(std::vector<int> &set_of_threads)
{
    int i;

    int num_threads = set_of_threads.size();

    printf("Number Threads: %d\n", num_threads);
    printf("Thread Pool Binding:");
    for (i = 0; i < num_threads; i++)
    {
        printf(" %d", set_of_threads[i]);
    }
    printf("\n");

    // set table head
    vector<string> ti;
    ti.resize(3);
    ti[0] = "Instruction Set";
    ti[1] = "Data Type";
    ti[2] = "Peak Performance";
    
    Table table;
    table.setColumnNum(3);
    table.addOneItem(ti);

    // set thread pool
    smtl_handle sh;
	smtl_init(&sh, set_of_threads);

    // traverse fp bench list
    for (i = 0; i < fp_bench_list.size(); i++)
    {
        pe_execute_bench(sh, PEAK_FP, &fp_bench_list[i], table);
    }
    table.print();

    Table table2;
    ti[0] = "Instruction Set";
    ti[1] = "Data Type";
    ti[2] = "Peak Bandwidth";
    table2.setColumnNum(3);
    table2.addOneItem(ti);
    // traverse mem bench list
    for(i = 0; i < mem_bench_list.size(); i++) {
        pe_execute_bench(sh, PEAK_MEM, &mem_bench_list[i], table2);
    }
    table2.print();
    
    smtl_fini(sh);
}

