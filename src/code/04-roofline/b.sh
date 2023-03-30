#!/bin/bash
#ins_set test
g++ -O3 -c table.cpp
g++ -O3 -c smtl.cpp
g++ -O3 -c cpubm_x86.cpp
g++ -O3 -c cpufp_x86.cpp
ins_set=`lscpu | grep Flags`
compile_commands=""

link_sources="g++ -pthread -O3 -o cpufp table.o smtl.o cpubm_x86.o cpufp_x86.o"
#avx instruction set check
if [[ $ins_set =~ "avx2" ]];
then
  echo "avx2 supported"
  compile_commands=${compile_commands}"g++ -c asm/cpufp_kernel_x86_avx.S;"
  link_sources=${link_sources}"cpufp_kernel_x86_avx.o"

fi

#avx512 instruction set check
if [[ $ins_set =~ "avx512" ]];
then
  echo "avx512 supported"
  compile_commands=${compile_commands}"g++ -c asm/cpufp_kernel_x86_avx512f.S;"
  link_sources=${link_sources}" cpufp_kernel_x86_avx512f.o"
fi

#avx512_vnni instruction set check
if [[ $ins_set =~ "avx512_vnni" ]];
then
  echo "avx512vnni supported"
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

eval ${compile_commands}
eval ${link_sources}$

#for file in `ls .`; do
#  echo $file
#done
