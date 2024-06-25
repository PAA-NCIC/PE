#include<string.h>


struct cpu_fp_x86{
  std::string isa;
  std::string type;
  std::string dim;
  int64_t num_loops;
  int64_t flops_per_loop;
  void (*bench)(int64_t);
};

struct cpu_mem_x86{
  std::string isa;
  std::string type;
  std::string dim;
  int32_t num_loops;
  int32_t dv_per_loop;
  void (*bench)(int64_t, void *);
  void *src1;
};

typedef enum{
  PEAK_FP,
  PEAK_MEM,
  CUSTOM
}BENCH_TYPE;
