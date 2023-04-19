#include "table.hpp"
#include "smtl.hpp"
#include "cpubm_x86.hpp"

#include <cstdint>
#include <ctime>
#include <vector>
#include <sstream>
#include <iomanip>
using namespace std;

static double get_time(struct timespec *start,
	struct timespec *end)
{
	return end->tv_sec - start->tv_sec +
		(end->tv_nsec - start->tv_nsec) * 1e-9;
}

typedef struct
{
    std::string isa;
    std::string type;
    std::string dim;
    int64_t num_loops;
    int64_t flops_per_loop;
    void (*bench)(int64_t);
} cpu_fp_x86;

typedef struct
{
    std::string isa;
    std::string type;
    std::string dim;
    int64_t num_loops;
    int64_t data_volum_per_loop;
    void *src1;             
    void *src2;     
    void *src3;
    void (*bench)(int64_t, void *, void *, void *);
} cpu_mem_x86;

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
    int64_t num_loops,
    int64_t data_volume_per_loop,
    void (*bench)(int64_t, void *, void*, void *))
{
    //peak mem
    cpu_mem_x86 mem_bench_item;
    mem_bench_item.isa = isa;
    mem_bench_item.type = type;
    mem_bench_item.num_loops = num_loops;
    mem_bench_item.data_volum_per_loop = data_volume_per_loop;
    mem_bench_list.push_back(mem_bench_item);
}

static void thread_bench_mem(void *params)
{
    cpu_mem_x86 *bm = (cpu_mem_x86 *)params;
    bm->bench(bm->num_loops, bm->src1, bm->src2, bm->src3);
}

static void pe_execute_fp(smtl_handle sh,
    cpu_fp_x86 &item,
    Table &table)
{
    struct timespec start, end;
    double time_used, perf;

    int i;
    int num_threads = smtl_num_threads(sh);

	// warm up
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_bench_fp, (void*)&item);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_bench_fp, (void*)&item);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	time_used = get_time(&start, &end);
	perf = item.num_loops * item.flops_per_loop * num_threads /
        time_used * 1e-9;
    
    stringstream ss;
    ss << std::setprecision(5) << perf << " " << item.dim;

    vector<string> cont;
    cont.resize(3);
    cont[0] = item.isa;
    cont[1] = item.type;
    cont[2] = ss.str();
    table.addOneItem(cont);
}

static void pe_execute_mem(smtl_handle sh,
    cpu_mem_x86 &item,
    Table &table)
{
    struct timespec start, end;
    double time_used, perf;

    int i;
    int num_threads = smtl_num_threads(sh);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_bench_mem, (void*)&item);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	time_used = get_time(&start, &end);
	perf = item.num_loops * item.data_volum_per_loop * num_threads /
        time_used * 1e-9;
    
    stringstream ss;
    ss << std::setprecision(5) << perf << " " << item.dim;

    vector<string> cont;
    cont.resize(3);
    cont[0] = item.isa;
    cont[1] = item.type;
    cont[2] = ss.str();
    table.addOneItem(cont);
}

void cpubm_do_bench(std::vector<int> &set_of_threads)
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
        pe_execute_fp(sh, fp_bench_list[i], table);
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
        pe_execute_mem(sh, mem_bench_list[i], table2);
    }
    table2.print();
    
    smtl_fini(sh);
}

