#include "../include/pe_jacobi.hpp"
#include "../../unity/include/help.hpp"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>

using namespace std;

const uint64_t KMAX = 10000001;
const uint64_t JMAX = 100;
const int REPEAT = 50;


int main(int argc, char *argv[]) {
  struct timespec start, end;
  double time_used, flops, perf_flops, perf_lup;
  
  double *A, *B;
  //aligned to 64 bytes 
  A = (double *)aligned_alloc(64, JMAX * KMAX * sizeof(double));
  B = (double *)aligned_alloc(64, JMAX * KMAX * sizeof(double));

  cout << "init data......" << endl;
  init_2d_array(A, JMAX, KMAX);
  init_2d_array(B, JMAX, KMAX);
  cout << "init data finished" << endl;
  if(argc >= 2) {
    if(!strcmp("2d", argv[1])) {
      if(!strcmp("noblock", argv[2])) {
        cout << "2d noblock test" << std::endl;
        cout << setw(12) << "jamx" << setw(12) << "kamx" << setw(12) << "lup"  << endl; 
        for(uint64_t kmax = 128; kmax < KMAX; kmax = kmax * 2) {
          //warm up
          pe_jacobi2d_blocking(A, B, JMAX, KMAX, 1.0, kmax);
          clock_gettime(CLOCK_MONOTONIC_RAW, &start);
          //call
          for(int r = 0; r < REPEAT; r++)
            pe_jacobi2d_blocking(A, B, JMAX, kmax, 1.0, kmax);
          clock_gettime(CLOCK_MONOTONIC_RAW, &end);

          time_used = get_time(&start, &end);
          flops = 1.0 * (JMAX - 2) * (kmax - 2) * 4;
          perf_flops = REPEAT *  flops / time_used * 1e-9;
          perf_lup = perf_flops / 4 ;
          cout << setw(12) << JMAX << setw(12) << kmax << setw(12)
              << setprecision(3) << perf_lup << setw(12) << endl;
        }
      }
      else if(!strcmp("block", argv[2])) {
        cout << setw(12) << "jamx" << setw(12) << "kamx" << setw(12) << "kblock" << setw(12) << "lup" << endl;
        uint32_t cache_size = (uint32_t)(atof(argv[3]) * 1024 * 1024);
        for(uint64_t kmax = 128; kmax < KMAX; kmax = kmax * 2) {
          uint32_t kblock = cache_size / 2 / 24 / kmax;
          kblock =  kmax < kblock ? kmax : kblock;
          //warm up
          pe_jacobi2d_blocking(A, B, JMAX, KMAX, 1.0, kblock);
          clock_gettime(CLOCK_MONOTONIC_RAW, &start);
          //call
          for(int r = 0; r < REPEAT; r++)
            pe_jacobi2d_blocking(A, B, JMAX, kmax, 1.0, kblock);
          clock_gettime(CLOCK_MONOTONIC_RAW, &end);

          time_used = get_time(&start, &end);
          flops = 1.0 * (JMAX - 2) * (kmax - 2) * 4;
          perf_lup = REPEAT *  flops / time_used * 1e-9 / 4 ;
          cout << setw(12) << JMAX << setw(12) << kmax << setw(12)
              << setprecision(3) << kblock << setw(12)
              << setprecision(3) << perf_lup << endl << endl;
        }
      }
    }     
    else if(!strcmp("3d", argv[1])) {
      if(!strcmp("noblock", argv[2])){
        cout << setw(12) << "kmax" << setw(12) << "lup" << endl;
        for(uint64_t kmax = 100; kmax < 451; kmax = kmax + 10) {
          //warm up
          //omp_set_num_threads(threads[t]);
          pe_jacobi3d_iparallel(A, B, kmax-1 , kmax-1, kmax-1, 1.0/6);
          clock_gettime(CLOCK_MONOTONIC_RAW, &start);
          for(int r = 0; r < REPEAT; r++)
            pe_jacobi3d_iparallel(A, B, kmax-1 , kmax-1, kmax-1, 1.0/6);
          clock_gettime(CLOCK_MONOTONIC_RAW, &end);

          time_used = get_time(&start, &end);
          flops = 1.0 * (kmax - 2) * (kmax - 2) * (kmax - 2) * 6;
          perf_lup = REPEAT *  flops / time_used * 1e-9 / 6;
          cout << setw(12) << kmax << setw(12) << setprecision(3) 
               << perf_lup << endl;
        }
      } 
      else if(!strcmp("block", argv[2])) {
        //cout << "3d test Cache size" << std::endl;
        uint32_t cache_size = (uint32_t)(atof(argv[3])* 1024 * 1024);
        cout << "cache size: " << atof(argv[3]) << "MB" << std::endl;
        cout << setw(12) << "i(j/k)max" << setw(12) << "jblock"
             << setw(12) << "lup" << endl;
        cache_size = cache_size & 0xFFFFFFE0; 
        for(uint64_t kmax = 100; kmax < 451; kmax = kmax + 10) {
          uint32_t jblock = cache_size / 16 / 2 / 24 / kmax;
          //omp_set_num_threads(16);
          pe_jacobi3d_iparallel_block(A, B, kmax -1 , kmax - 1, kmax - 1, 1.0, jblock);  //
          clock_gettime(CLOCK_MONOTONIC_RAW, &start);
          for(int r = 0; r < REPEAT; r++)
            pe_jacobi3d_iparallel_block(A, B, kmax -1 , kmax - 1, kmax - 1, 1.0, jblock);  //
          clock_gettime(CLOCK_MONOTONIC_RAW, &end);

          time_used = get_time(&start, &end);
          flops = 1.0 * (kmax - 2) * (kmax - 2) * (kmax - 2) * 6;
          perf_lup = REPEAT *  flops / time_used * 1e-9 / 6;
          cout << setw(12) << kmax << setw(12) << jblock << setw(12) 
                << setprecision(3) << perf_lup << endl;
        }
      } 
      else if(!strcmp("jblock", argv[2])) {
        cout << setw(12) << "jblock" << setw(12) << "i(j/k)max" << setw(12) << "lup" << endl;
        for(uint32_t jblock = 8; jblock < 256; jblock += 8) {
          uint32_t kmax = 400;
          //omp_set_num_threads(16);
          clock_gettime(CLOCK_MONOTONIC_RAW, &start);
          for(int r = 0; r < REPEAT; r++)
            pe_jacobi3d_iparallel_block(A, B, kmax -1 , kmax - 1, kmax - 1, 1.0, jblock);  //
          clock_gettime(CLOCK_MONOTONIC_RAW, &end);

          time_used = get_time(&start, &end);
          flops = 1.0 * (kmax - 2) * (kmax - 2) * (kmax - 2) * 6;
          perf_lup = REPEAT *  flops / time_used * 1e-9 / 6;
          cout << setw(12) << jblock << setw(12) << kmax << setw(12) 
                << setprecision(3) << perf_lup << setw(12) << endl;
        }
      }
      else if(!strcmp("jparallel", argv[2])) {
        cout << setw(12) << "kmax" << setw(12) << "lup" << setw(12) << endl;
        for(uint64_t kmax = 100; kmax < 1000; kmax = kmax + 40) {
          //warm up
          pe_jacobi3d_jparallel(A, B, kmax, kmax, kmax, 1.0);
          clock_gettime(CLOCK_MONOTONIC_RAW, &start);
          //call
          for(int r = 0; r < REPEAT; r++)
            pe_jacobi3d_jparallel(A, B, kmax, kmax, kmax, 1.0);
          clock_gettime(CLOCK_MONOTONIC_RAW, &end);

          time_used = get_time(&start, &end);
          flops = 1.0 * (kmax - 2) * (kmax - 2) * (kmax - 2) * 6;
          perf_flops = REPEAT *  flops / time_used * 1e-9;
          perf_lup = perf_flops / 6 ;
          cout << setw(12) << kmax << setw(12) <<setprecision(3) << perf_lup << endl;
        }
      }
      else if(!strcmp("ntstore", argv[2])) {
      //A is aligned to 64 bytes, A_7 + 1 shouled be aligned to 64 bytes
      //we do this for convient, since nt store require addr is aligned to 64 bytes;
      //we assume that 
      double* A_7 = A + 7;  //A + 7;
      // double* B_8 = B + 7; //B + 7;
      // cout << "A:" << A << "\t B:" << B <<  endl;
      // cout << "A_8:" << A_8 << "\t B_8:" << B_8 << endl;
      // cout << "3d test i parallel NT-store" << std::endl;
      cout << setw(12) << "i(j/k)max" << setw(12)  << "lup" << endl;
      uint32_t jb = 48;
      for(uint32_t kmax = 128; kmax < 1000; kmax = kmax + 64) {
        pe_jacobi3d_iparallel_block_ntstore512(A_7, B, kmax, kmax, kmax, 1.0, jb);

        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        //call
        for(int r = 0; r < REPEAT; r++)
          pe_jacobi3d_iparallel_block_ntstore512(A_7, B, kmax, kmax, kmax, 1.0, jb);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        time_used = get_time(&start, &end);
        //since for aligned purpose, we only compute 64-aligned parts
        //we pad k for 6 elements
        flops = 1.0 * (kmax - 2) * (kmax - 2) * (kmax - 8) * 6;
        perf_lup = REPEAT *  flops / time_used * 1e-9 / 6 ;
        cout << setw(12) << kmax << setw(12) << setprecision(3) << perf_lup << endl;
     }
    } 
    }
  }    
  free(A);
  free(B);
  return 0;
}
