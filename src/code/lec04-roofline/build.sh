#!/bin/bash
#ins_set test
g++ -g -O3 -c table.cpp
g++ -g -O3 -c smtl.cpp
ins_set=`lscpu | grep Flags`
compile_commands=""

link_sources="g++ -pthread -O0 -o pe_bench table.o smtl.o cpubm_x86.o pe_bench.o"
inst_flags=""

if [[ $ins_set =~ "sse" ]];
then
  echo "sse supported"
  inst_flags=${inst_flags}" -DSSE" 
  compile_commands=${compile_commands}"g++ -c -O0 asm/cpufp_kernel_x86_sse.S;"
  link_sources=${link_sources}" cpufp_kernel_x86_sse.o"
  #mem
  compile_commands=${compile_commands}"g++ -c -O0 asm/load_kernel_x86_sse.S;"
  link_sources=${link_sources}" load_kernel_x86_sse.o"
fi

if [[ $ins_set =~ "fma" ]];
then
  echo "fma supported"
  inst_flags=${inst_flags}" -DFMA" 
  compile_commands=${compile_commands}"g++ -c -O0 asm/cpufp_kernel_x86_fma.S;"
  link_sources=${link_sources}" cpufp_kernel_x86_fma.o"
fi

#avx instruction set check
if [[ $ins_set =~ "avx2" ]];
then
  echo "avx2 supported"
  inst_flags=${inst_flags}" -DAVX" 
  compile_commands=${compile_commands}"g++ -c -O0 asm/cpufp_kernel_x86_avx.S;"
  link_sources=${link_sources}" cpufp_kernel_x86_avx.o"
  #mem
  compile_commands=${compile_commands}"g++ -c -O0 asm/load_kernel_x86_avx.S;"
  link_sources=${link_sources}" load_kernel_x86_avx.o"
fi

#avx512 instruction set check
if [[ $ins_set =~ "avx512" ]];
then
  echo "avx512 supported"
  inst_flags=${inst_flags}" -DAVX512" 
  compile_commands=${compile_commands}"g++ -c -O0 asm/cpufp_kernel_x86_avx512f.S;"
  link_sources=${link_sources}" cpufp_kernel_x86_avx512f.o"
  #mem
  compile_commands=${compile_commands}"g++ -c -O0 asm/load_kernel_x86_avx512.S;"
  link_sources=${link_sources}" load_kernel_x86_avx512.o"
fi

#avx512_vnni instruction set check
if [[ $ins_set =~ "avx512_vnni" ]];
then
  echo "avx512vnni supported"
  inst_flags=${inst_flags}" -DAVX512_VNNI" 
  compile_commands=${compile_commands}"g++ -c -O0 asm/cpufp_kernel_x86_avx512_vnni.S;"
  link_sources=${link_sources}" cpufp_kernel_x86_avx512_vnni.o"
fi

#avx_vnni instruction set check
if [[ $ins_set =~ "avx_vnni" ]];
then
  echo "avx_vnni supported"
  inst_flags=${inst_flags}" -DAVX_VNNI"
  compile_commands=${compile_commands}"g++ -c -O0 asm/cpufp_kernel_x86_avx_vnni.S;"
  link_sources=${link_sources}" cpufp_kernel_x86_avx_vnni.o"
fi

echo ${inst_flags}
g++ -g -O2 -c cpubm_x86.cpp ${inst_flags}
g++ -g -O2 -c pe_bench.cpp ${inst_flags}
echo ${compile_commands}
eval ${compile_commands}
echo ${link_sources}
eval ${link_sources}

#clean
rm *.o

#for file in `ls .`; do
#  echo $file
#done
