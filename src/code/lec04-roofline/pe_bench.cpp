#include "cpubm_x86.hpp"

#include <cstring>
#include <cstdint>
#include <vector>
using namespace std;

extern "C"
{
void cpufp_kernel_x86_sse_fp32(int64_t);
void cpufp_kernel_x86_sse_fp64(int64_t);

void cpufp_kernel_x86_avx_fp32(int64_t);
void cpufp_kernel_x86_avx_fp64(int64_t);

void cpufp_kernel_x86_fma_fp32(int64_t);
void cpufp_kernel_x86_fma_fp64(int64_t);

void cpufp_kernel_x86_avx512f_fp32(int64_t);
void cpufp_kernel_x86_avx512f_fp64(int64_t);

void cpufp_kernel_x86_avx512_vnni_int8(int64_t);
void cpufp_kernel_x86_avx512_vnni_int16(int64_t);

void cpufp_kernel_x86_avx_vnni_int8(int64_t);
void cpufp_kernel_x86_avx_vnni_int16(int64_t);

/**********************************************
 * size_byte: the size of memory to load in bytes
 * src: memory ptr
*/
void load_kernel_x86_avx512(int64_t size_byte, void *src);
void load_kernel_x86_avx(int64_t size_byte, void *src);
void load_kernel_x86_sse(int64_t size_byte, void *src);

}

static void parse_thread_pool(char *sets,
    vector<int> &set_of_threads)
{
    if (sets[0] != '[')
    {
        return;
    }
    int pos = 1;
    int left = 0, right = 0;
    int state = 0;
    while (sets[pos] != ']' && sets[pos] != '\0')
    {
        if (state == 0)
        {
            if (sets[pos] >= '0' && sets[pos] <= '9')
            {
                left *= 10;
                left += (int)(sets[pos] - '0');
            }
            else if (sets[pos] == ',')
            {
                set_of_threads.push_back(left);
                left = 0;
            }
            else if (sets[pos] == '-')
            {
                right = 0;
                state = 1;
            }
        }
        else if (state == 1)
        {
            if (sets[pos] >= '0' && sets[pos] <= '9')
            {
                right *= 10;
                right += (int)(sets[pos] - '0');
            }
            else if (sets[pos] == ',')
            {
                int i;
                for (i = left; i <= right; i++)
                {
                    set_of_threads.push_back(i);
                }
                left = 0;
                state = 0;
            }
        }
        pos++;
    }
    if (sets[pos] != ']')
    {
        return;
    }
    if (state == 0)
    {
        set_of_threads.push_back(left);
    }
    else if (state == 1)
    {
        int i;
        for (i = left; i <= right; i++)
        {
            set_of_threads.push_back(i);
        }
    }
}

static void register_isa()
{
#ifdef AVX512_VNNI
    reg_new_fp_bench("AVX512_VNNI", "INT8", "GFLOPS",
        0x20000000LL, 1280LL,
        cpufp_kernel_x86_avx512_vnni_int8);
    reg_new_fp_bench("AVX512_VNNI", "INT16", "GFLOPS",
        0x20000000LL, 640LL,
        cpufp_kernel_x86_avx512_vnni_int16);
#endif

#ifdef AVX512
    reg_new_fp_bench("AVX512F", "FP32", "GFLOPS",
        0x20000000LL, 320LL,
        cpufp_kernel_x86_avx512f_fp32);
    reg_new_fp_bench("AVX512F", "FP64", "GFLOPS",
        0x20000000LL, 160LL,
        cpufp_kernel_x86_avx512f_fp64);
    reg_new_mem_bench("AVX512", "load A[i]", "GB/s",
        50, 1024*1024*32, load_kernel_x86_avx512);
#endif

#ifdef AVX_VNNI
    reg_new_fp_bench("AVX_VNNI", "INT8", "GFLOPS",
        0x40000000LL, 640LL,
        cpufp_kernel_x86_avx_vnni_int8);
    reg_new_fp_bench("AVX_VNNI", "INT16", "GFLOPS",
        0x40000000LL, 320LL,
        cpufp_kernel_x86_avx_vnni_int16);
#endif

#ifdef AVX
    reg_new_fp_bench("AVX", "FP32", "GFLOPS",
        0x40000000LL, 96LL,
        cpufp_kernel_x86_avx_fp32);
    reg_new_fp_bench("AVX", "FP64", "GFLOPS",
        0x40000000LL, 48LL,
        cpufp_kernel_x86_avx_fp64);
    reg_new_mem_bench("AVX", "load A[i]", "GB/s",
        50, 1024*1024*32, load_kernel_x86_avx);
#endif

#ifdef FMA
    reg_new_fp_bench("FMA", "FP32", "GFLOPS",
        0x80000000LL, 160LL,
        cpufp_kernel_x86_fma_fp32);
    reg_new_fp_bench("FMA", "FP64", "GFLOPS",
        0x80000000LL, 80LL,
        cpufp_kernel_x86_fma_fp64);
#endif

#ifdef SSE
    reg_new_fp_bench("SSE", "FP32", "GFLOPS",
        0x80000000LL, 64LL,
        cpufp_kernel_x86_sse_fp32);
    reg_new_fp_bench("SSE", "FP64", "GFLOPS",
        0x80000000LL, 32LL,
        cpufp_kernel_x86_sse_fp64);
    reg_new_mem_bench("SSE", "load A[i]", "GB/s",
        50, 1024*1024*32, load_kernel_x86_sse);
#endif
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s --thread_pool=[xxx]\n", argv[0]);
        fprintf(stderr, "[xxx] indicates all cores to benchmark.\n");
        fprintf(stderr, "Example: [0,3,5-8,13-15].\n");
        fprintf(stderr, "Notice: there must NOT be any spaces.\n");
        exit(0);
    }

    if (strncmp(argv[1], "--thread_pool=", 14))
    {
        fprintf(stderr, "Error: You must set --thread_pool parameter.\n");
        fprintf(stderr, "Usage: %s --thread_pool=[xxx]\n", argv[0]);
        fprintf(stderr, "[xxx] indicates all cores to benchmark.\n");
        fprintf(stderr, "Example: [0,3,5-8,13-15].\n");
        fprintf(stderr, "Notice: there must NOT be any spaces.\n");
        exit(0);
    }

    vector<int> set_of_threads;

    parse_thread_pool(argv[1] + 14, set_of_threads);

    register_isa();
    pe_bench(set_of_threads);

    return 0;
}

