#!/bin/bash
#ins_set test
g++ -O3 -c table.cpp
g++ -O3 -c smtl.cpp
ins_set=`lscpu | grep Flags`
compile_commands=""

link_sources="g++ -pthread -O3 -o pe_bench table.o smtl.o cpubm_x86.o pe_bench.o"
inst_flags=""
#avx instruction set check
if [[ $ins_set =~ "avx2" ]];
then
  echo "avx2 supported"
  inst_flags=${inst_flags}" -DAVX" 
  compile_commands=${compile_commands}"g++ -c asm/cpufp_kernel_x86_avx.S;"
  link_sources=${link_sources}" cpufp_kernel_x86_avx.o"
fi

#avx512 instruction set check
if [[ $ins_set =~ "avx512" ]];
then
  echo "avx512 supported"
  inst_flags=${inst_flags}" -DAVX512" 
  compile_commands=${compile_commands}"g++ -c asm/cpufp_kernel_x86_avx512f.S;"
  link_sources=${link_sources}" cpufp_kernel_x86_avx512f.o"
  #mem
  compile_commands=${compile_commands}"g++ -c asm/pe_kernel_x86_avx512_vector_copy.S;"
  link_sources=${link_sources}" pe_kernel_x86_avx512_vector_copy.o"
fi

#avx512_vnni instruction set check
if [[ $ins_set =~ "avx512_vnni" ]];
then
  echo "avx512vnni supported"
  inst_flags=${inst_flags}" -DAVX512_VNNI" 
  compile_commands=${compile_commands}"g++ -c asm/cpufp_kernel_x86_avx512_vnni.S;"
  link_sources=${link_sources}" cpufp_kernel_x86_avx512_vnni.o"
fi

#avx_vnni instruction set check
if [[ $ins_set =~ "avx_vnni" ]];
then
  echo "avx_vnni supported"
  compile_commands=${compile_commands}"echo avx_vnni;"
  link_sources=${link_sources}" cpufp_kernel_x86_avx_vnni.o"
fi

echo ${inst_flags}
g++ -O3 -c cpubm_x86.cpp ${inst_flags}
g++ -O3 -c pe_bench.cpp ${inst_flags}
eval ${compile_commands}
echo ${link_sources}
eval ${link_sources}

#clean
rm *.o

#for file in `ls .`; do
#  echo $file
#done
