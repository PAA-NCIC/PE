#include "../include/pe_jacobi.hpp"
#include "../include/help.hpp"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>

using namespace std;

const uint64_t KMAX = 10000001;
const uint64_t JMAX = 100;
const int REPEAT = 1;


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
  if(argc == 2) {
    if(!strcmp("2dblock", argv[1])) {
    /*  cout << "kblock == kmax test" << std::endl;
      cout << setw(12) << "jamx" << setw(12) << "kamx" << setw(12) << "lup" << setw(12) << "flops" << endl; 
      for(uint64_t kmax = 128; kmax < KMAX; kmax = kmax * 2) {
        //warm up
        //pe_jacobi2d_blocking(A, B, JMAX, KMAX, 1.0, kblock);
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
          pe_jacobi2d_blocking(A, B, JMAX, kmax, 1.0, kblock);
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
          pe_jacobi2d_blocking(A, B, JMAX, kmax, 1.0, kblock);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        time_used = get_time(&start, &end);
        flops = 1.0 * (JMAX - 2) * (kmax - 2) * 4;
        perf_flops = REPEAT *  flops / time_used * 1e-9;
        perf_lup = perf_flops / 4 ;
        cout << setw(12) << JMAX << setw(12) << kmax << setw(12)
             << setprecision(3) << perf_lup << setw(12)
             << setprecision(3) << perf_flops << endl << endl;
      }*/

      cout << "roofline validate" << endl;
      int typical_kbs[4] = {256, 21000, 462000, 1000000};
      cout << setw(12) << "threads" << setw(12) << "jamx" << setw(12) << "kb" << setw(12) << "lup" << setw(12) << "flops" << endl;
      for(int i = 0; i < 4; i++) {
        int kblock = typical_kbs[i];
        //roofline validate for each typical kbs
        for(int threads = 1; threads <=16; threads += 2) {
          omp_set_num_threads(threads);
          clock_gettime(CLOCK_MONOTONIC_RAW, &start);
          for(int r = 0; r < REPEAT; r++)
            pe_jacobi2d_blocking(A, B, JMAX, KMAX, 1.0, kblock);
          clock_gettime(CLOCK_MONOTONIC_RAW, &end);
          time_used = get_time(&start, &end);
          flops = 1.0 * (JMAX - 2) * (KMAX - 2) * 4;
          perf_flops = REPEAT *  flops / time_used * 1e-9;
          perf_lup = perf_flops / 4 ;
          cout << setw(12) << threads << setw(12) << JMAX 
            << setw(12) << kblock << setw(12)
            << setprecision(3) << perf_lup << setw(12)
            << setprecision(3) << perf_flops << endl << endl;
        }      
      }
    } else if(!strcmp("3dblock", argv[1])) {
      int threads[5] = {1, 4, 8, 12, 16};
      /*cout << "3d test i parallel" << std::endl;
      cout << setw(12) << "threads" << setw(12) << "imax" << setw(12) << "jamx" << setw(12) << setw(12) << "kmax" << setw(12) << "lup" << setw(12) << "flops" << endl;
      for(int t = 0; t < 5; t++) {
        for(uint64_t kmax = 100; kmax < 1000; kmax = kmax + 40) {
          //warm up
          //pe_jacobi2d_blocking(A, B, JMAX, KMAX, 1.0, kblock);
          uint64_t imax = JMAX * KMAX / (kmax * kmax);
          omp_set_num_threads(threads[t]);
          clock_gettime(CLOCK_MONOTONIC_RAW, &start);
          //call
          for(int r = 0; r < REPEAT; r++)
            pe_jacobi3d_iparallel(A, B, imax -1 , kmax - 1, kmax - 1, 1.0);
          clock_gettime(CLOCK_MONOTONIC_RAW, &end);

          time_used = get_time(&start, &end);
          flops = 1.0 * (imax - 2) * (kmax - 2) * (kmax - 2) * 6;
          perf_flops = REPEAT *  flops / time_used * 1e-9;
          perf_lup = perf_flops / 6 ;
          cout << setw(12) << threads[t] << setw(12) << imax << setw(12) 
              << kmax << setw(12) << kmax << setw(12) <<setprecision(3) 
              << perf_lup << setw(12) << setprecision(3) << perf_flops << endl;
        }
      }*/
      /*
      cout << "3d test Cache size" << std::endl;
      cout << setw(12) << "jblock" << setw(12) << "i(j/k)max" << setw(12) << "lup" << endl;
      uint32_t CS[4] = {2 * 1024 * 1024, 10 * 1024 * 1024, 22 * 1024 * 1024, 1 << 30}; //2MB, 10MB, 22MB, inf
      for(int i = 0; i < 4; i++) {
        for(uint64_t kmax = 100; kmax < 601; kmax = kmax + 50) {
          uint32_t jblock = CS[i] / 2 / 16 / 3 / kmax / 8;
          //uint64_t imax = JMAX * KMAX / (kmax * kmax);
          omp_set_num_threads(16);
          clock_gettime(CLOCK_MONOTONIC_RAW, &start);
          for(int r = 0; r < REPEAT; r++)
            pe_jacobi3d_iparallel_block(A, B, kmax -1 , kmax - 1, kmax - 1, 1.0, jblock);  //
          clock_gettime(CLOCK_MONOTONIC_RAW, &end);

          time_used = get_time(&start, &end);
          flops = 1.0 * (kmax - 2) * (kmax - 2) * (kmax - 2) * 6;
          perf_flops = REPEAT *  flops / time_used * 1e-9;
          perf_lup = perf_flops / 6;
          cout << setw(12) << jblock << setw(12) << kmax << setw(12) 
                << setprecision(3) << perf_lup << setw(12) << endl;
        }
      }
      */
     /*
      cout << setw(12) << "jblock" << setw(12) << "i(j/k)max" << setw(12) << "lup" << endl;
      for(uint32_t jblock = 8; jblock < 256; jblock += 8) {
        uint32_t kmax = 400;
        omp_set_num_threads(16);
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        for(int r = 0; r < REPEAT; r++)
          pe_jacobi3d_iparallel_block(A, B, kmax -1 , kmax - 1, kmax - 1, 1.0, jblock);  //
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        time_used = get_time(&start, &end);
        flops = 1.0 * (kmax - 2) * (kmax - 2) * (kmax - 2) * 6;
        perf_flops = REPEAT *  flops / time_used * 1e-9;
        perf_lup = perf_flops / 6;
        cout << setw(12) << jblock << setw(12) << kmax << setw(12) 
              << setprecision(3) << perf_lup << setw(12) << endl;
      }
*/
      /*for(int jblock = 20 ; jblock <=320 ; jblock += 20) {
        for(uint64_t kmax = 100; kmax < 601; kmax = kmax + 50) {
          //uint64_t imax = JMAX * KMAX / (kmax * kmax);
          omp_set_num_threads(16);
          clock_gettime(CLOCK_MONOTONIC_RAW, &start);
          //call
          for(int r = 0; r < REPEAT; r++)
            pe_jacobi3d_iparallel_block(A, B, kmax -1 , kmax - 1, kmax - 1, 1.0, jblock);
          clock_gettime(CLOCK_MONOTONIC_RAW, &end);

          time_used = get_time(&start, &end);
          flops = 1.0 * (kmax - 2) * (kmax - 2) * (kmax - 2) * 6;
          perf_flops = REPEAT *  flops / time_used * 1e-9;
          perf_lup = perf_flops / 6;
          cout << setw(12) << jblock << setw(12) << kmax << setw(12) 
               << setprecision(3) << perf_lup << setw(12) << endl;
        }
      }
      */
/*
      cout << "3d test j parallel" << std::endl;
      cout << setw(12) << "threads" << setw(12) << "imax" << setw(12) << "jamx" << setw(12) << setw(12) << "kmax" << setw(12) << "lup" << setw(12) << "flops" << endl;
      for(int t = 0; t < 5; t++) {
        for(uint64_t kmax = 100; kmax < 1000; kmax = kmax + 40) {
          //warm up
          //pe_jacobi2d_blocking(A, B, JMAX, KMAX, 1.0, kblock);
          uint64_t imax = JMAX * KMAX / (kmax * kmax);
          omp_set_num_threads(threads[t]);
          clock_gettime(CLOCK_MONOTONIC_RAW, &start);
          //call
          for(int r = 0; r < REPEAT; r++)
            pe_jacobi3d_jparallel(A, B, imax, kmax, kmax, 1.0);
          clock_gettime(CLOCK_MONOTONIC_RAW, &end);

          time_used = get_time(&start, &end);
          flops = 1.0 * (imax - 2) * (kmax - 2) * (kmax - 2) * 6;
          perf_flops = REPEAT *  flops / time_used * 1e-9;
          perf_lup = perf_flops / 6 ;
          cout << setw(12) << threads[t] << setw(12) << imax << setw(12) << kmax 
              << setw(12) << kmax << setw(12) <<setprecision(3) << perf_lup 
              << setw(12) << setprecision(3) << perf_flops << endl;
        }
      }*/
       //A is aligned to 64 bytes, A_aligned shouled be aligned to 8 bytes, which can make sure A_8 + 1 is aligned to 16 bytes;
      double* A_8 = A + 7;  //A + 7;
      double* B_8 = B + 7; //B + 7;
      cout << "A:" << A << "\t B:" << B <<  endl;
      cout << "A_8:" << A_8 << "\t B_8:" << B_8 << endl;
      cout << "3d test i parallel NT-store" << std::endl;
      cout << setw(4) << "jb" << setw(12) << "i(j/k)max" << setw(12)  << "lup" << endl;
      for(int jb = 8; jb < 128; jb += 8) {
      for(uint32_t kmax = 104; kmax < 1000; kmax = kmax + 104) {
        //uint64_t imax = JMAX * KMAX / (kmax * kmax);
        omp_set_num_threads(1);
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        //call
        for(int r = 0; r < REPEAT; r++)
          //pe_jacobi3d_iparallel_block_ntstore(A_8, B_8, kmax, kmax, kmax, 1.0, jb);
          //pe_jacobi3d_iparallel_block_ntstore256(A_8, B_8, kmax, kmax, kmax, 1.0, jb);
          pe_jacobi3d_iparallel_block_ntstore512(A_8, B_8, kmax, kmax, kmax, 1.0, jb);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        time_used = get_time(&start, &end);
        flops = 1.0 * (kmax - 2) * (kmax - 2) * (kmax - 2) * 6;
        perf_flops = REPEAT *  flops / time_used * 1e-9;
        perf_lup = perf_flops / 6 ;
        cout << setw(4) << jb << setw(12) << kmax << setw(12) << setprecision(3) << perf_lup << endl;
      }
      /*for(uint32_t kmax = 100; kmax < 1000; kmax = kmax + 100) {
        //uint64_t imax = JMAX * KMAX / (kmax * kmax);
        omp_set_num_threads(16);
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        //call
        for(int r = 0; r < REPEAT; r++)
          pe_jacobi3d_iparallel_block(A, B, kmax, kmax, kmax, 1.0, jb);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        time_used = get_time(&start, &end);
        flops = 1.0 * (kmax - 2) * (kmax - 2) * (kmax - 2) * 6;
        perf_flops = REPEAT *  flops / time_used * 1e-9;
        perf_lup = perf_flops / 6 ;
        cout << setw(4) << jb << setw(12) << kmax << setw(12) << setprecision(3) << perf_lup << endl;
      }*/
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
