#include "../include/pe_jacobi.hpp"
#include "../include/help.hpp"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>

using namespace std;

const uint64_t KMAX = 17000000;
const uint64_t JMAX = 200;
const int REPEAT = 5;

int main(int argc, char *argv[]) {
  struct timespec start, end;
  double time_used, flops, perf_flops, perf_lup;
  //for simple
  double *A, *B, *A_aligned, *B_aligned;
  A = (double *)malloc(JMAX * KMAX * sizeof(double) + 64);
  B = (double *)malloc(JMAX * KMAX * sizeof(double) + 64);
  //aligned to 64 bytes manually
  A_aligned = (double*)(((uint64_t)A + 64) & 0xFFFFFFC0);
  B_aligned = (double*)(((uint64_t)B + 64) & 0xFFFFFFC0);
  cout << "init data......" << endl;
  init_2d_array(A, JMAX, KMAX);
  init_2d_array(B, JMAX, KMAX);
  cout << "init data finished" << endl;
  if(argc == 2) {
    if(!strcmp("2dblock", argv[1])) {
      cout << "kblock == kmax test" << std::endl;
      cout << setw(12) << "jamx" << setw(12) << "kamx" << setw(12) << "lup" << setw(12) << "flops" << endl;
      for(uint64_t kmax = 128; kmax < KMAX; kmax = kmax * 2) {
        //warm up
        //pe_jacobi2d_blocking(A, B, JMAX, KMAX, 1.0, kblock);
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        //call
        for(int r = 0; r < REPEAT; r++)
          pe_jacobi2d_blocking(A_aligned, B_aligned, JMAX, kmax, 1.0, kmax);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        time_used = get_time(&start, &end);
        flops = 1.0 * (JMAX - 2) * (kmax - 2) * 4;
        perf_flops = REPEAT *  flops / time_used * 1e-9;
        perf_lup = perf_flops / 4 ;
        cout << setw(12) << JMAX << setw(12) << kmax << setw(12)
             << setprecision(3) << perf_lup << setw(12)
             << setprecision(3) << perf_flops << endl;
      }

      cout << "kblock = max(kmax, 21000(L2 1MB)) test  " << std::endl;
      cout << setw(12) << "jamx" << setw(12) << "kamx" << setw(12) << "lup" << setw(12) << "flops" << endl;
       uint64_t L2_block = 21000;
      for(uint64_t kmax = 128; kmax < KMAX; kmax = kmax * 2) {
        //L2 chache 1MB
        unsigned int kblock =  kmax < L2_block ? kmax : L2_block;
        //warm up
        //pe_jacobi2d_blocking(A, B, JMAX, KMAX, 1.0, kblock);
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        //call
        for(int r = 0; r < REPEAT; r++)
          pe_jacobi2d_blocking(A_aligned, B_aligned, JMAX, kmax, 1.0, kblock);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        time_used = get_time(&start, &end);
        flops = 1.0 * (JMAX - 2) * (kmax - 2) * 4;
        perf_flops = REPEAT *  flops / time_used * 1e-9;
        perf_lup = perf_flops / 4 ;
        cout << setw(12) << JMAX << setw(12) << kmax << setw(12)
             << setprecision(3) << perf_lup << setw(12)
             << setprecision(3) << perf_flops << endl << endl;
      }
      cout << "kblock = max(kmax, 462000(L3 22MB)) test  " << std::endl;
      cout << setw(12) << "jamx" << setw(12) << "kamx" << setw(12) << "lup" << setw(12) << "flops" << endl;
      uint64_t L3_block = 462000;
      for(uint64_t kmax = 128; kmax < KMAX; kmax = kmax * 2) {
        //L2 chache 1MB
        unsigned int kblock =  kmax < L3_block ? kmax : L3_block;
        //warm up
        //pe_jacobi2d_blocking(A, B, JMAX, KMAX, 1.0, kblock);
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        //call
        for(int r = 0; r < REPEAT; r++)
          pe_jacobi2d_blocking(A_aligned, B_aligned, JMAX, kmax, 1.0, kblock);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        time_used = get_time(&start, &end);
        flops = 1.0 * (JMAX - 2) * (kmax - 2) * 4;
        perf_flops = REPEAT *  flops / time_used * 1e-9;
        perf_lup = perf_flops / 4 ;
        cout << setw(12) << JMAX << setw(12) << kmax << setw(12)
             << setprecision(3) << perf_lup << setw(12)
             << setprecision(3) << perf_flops << endl << endl;
      }
    } else if(!strcmp("3dblock", argv[1])) {
      cout << "3d test i parallel" << std::endl;
      cout << setw(12) << "imax" << setw(12) << "jamx" << setw(12) 
      << setw(12) << "kmax" << setw(12) << "lup" << setw(12) << "flops" << endl;
      for(uint64_t kmax = 100; kmax < 1000; kmax = kmax + 20) {
        //warm up
        //pe_jacobi2d_blocking(A, B, JMAX, KMAX, 1.0, kblock);
        uint64_t imax = JMAX * KMAX / (kmax * kmax);
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        //call
        for(int r = 0; r < REPEAT; r++)
          pe_jacobi3d_iparallel(A_aligned, B_aligned, imax, kmax, kmax, 1.0);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        time_used = get_time(&start, &end);
        flops = 1.0 * (imax - 2) * (kmax - 2) * (kmax - 2) * 4;
        perf_flops = REPEAT *  flops / time_used * 1e-9;
        perf_lup = perf_flops / 4 ;
        cout << setw(12) << imax << setw(12) << kmax << setw(12) 
             << kmax << setw(12) <<setprecision(3) << perf_lup 
             << setw(12) << setprecision(3) << perf_flops << endl << endl;
      }

      cout << "3d test j parallel" << std::endl;
      cout << setw(12) << "imax" << setw(12) << "jamx" << setw(12) 
      << setw(12) << "kmax" << setw(12) << "lup" << setw(12) << "flops" << endl;
      for(uint64_t kmax = 100; kmax < 1000; kmax = kmax + 20) {
        //warm up
        //pe_jacobi2d_blocking(A, B, JMAX, KMAX, 1.0, kblock);
        uint64_t imax = JMAX * KMAX / (kmax * kmax);
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        //call
        for(int r = 0; r < REPEAT; r++)
          pe_jacobi3d_jparallel(A_aligned, B_aligned, imax, kmax, kmax, 1.0);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        time_used = get_time(&start, &end);
        flops = 1.0 * (imax - 2) * (kmax - 2) * (kmax - 2) * 4;
        perf_flops = REPEAT *  flops / time_used * 1e-9;
        perf_lup = perf_flops / 4 ;
        cout << setw(12) << imax << setw(12) << kmax << setw(12) 
             << kmax << setw(12) <<setprecision(3) << perf_lup 
             << setw(12) << setprecision(3) << perf_flops << endl;
      }

      cout << "3d test i parallel NT-store" << std::endl;
      cout << setw(12) << "imax" << setw(12) << "jamx" << setw(12) 
      << setw(12) << "kmax" << setw(12) << "lup" << setw(12) << "flops" << endl;
      for(uint64_t kmax = 100; kmax < 1000; kmax = kmax + 20) {
        //warm up
        //pe_jacobi2d_blocking(A, B, JMAX, KMAX, 1.0, kblock);
        uint64_t imax = JMAX * KMAX / (kmax * kmax);
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        //call
        for(int r = 0; r < REPEAT; r++)
          pe_jacobi3d_iparallel_ntstore(A_aligned, B_aligned, imax, kmax, kmax, 1.0);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        time_used = get_time(&start, &end);
        flops = 1.0 * (imax - 2) * (kmax - 2) * (kmax - 2) * 4;
        perf_flops = REPEAT *  flops / time_used * 1e-9;
        perf_lup = perf_flops / 4 ;
        cout << setw(12) << imax << setw(12) << kmax << setw(12) 
             << kmax << setw(12) <<setprecision(3) << perf_lup 
             << setw(12) << setprecision(3) << perf_flops << endl << endl;
      }
    } else {
      cout << setw(10) << "kamx" << setw(10) << "lup" << setw(10) << "flops" << endl;
      for(uint64_t kmax = 128; kmax < KMAX; kmax = kmax * 2) {
        //unsigned int kblock =  k < 53333 ? k : 53333;
        uint64_t jmax = (uint64_t)(1.0 * JMAX * KMAX / kmax);
        //warm up
        pe_jacobi2d(A, B, jmax, kmax, 1.0, 1);

        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        //call
        pe_jacobi2d(A, B, jmax+1, kmax+1, 1., 1);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        time_used = get_time(&start, &end);
        flops = (jmax - 2) * (kmax - 2) * 4;
        perf_flops = flops / time_used * 1e-9;
        perf_lup = perf_flops / 4 ;
        cout << setw(10) << jmax << setw(10) << kmax << setw(10)
             << setprecision(2) << perf_lup << setw(10)
             << setprecision(2) << perf_flops << endl;
      }
    }
  }
  free(A);
  free(B);
  return 0;
}
