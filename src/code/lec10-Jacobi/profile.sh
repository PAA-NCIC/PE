#!/bin/bash
#print table head

########## bw test ###########
# rm res -rf
# echo "threads                           " >> res
# echo "Memory read bandwidth [MBytes/s]  " >> res
# echo "Memory read data volume [GBytes]  " >> res
# echo "Memory write bandwidth [MBytes/s] " >> res
# echo "Memory write data volume [GBytes] " >> res
# echo "Memory bandwidth [MBytes/s]       " >> res
# echo "Memory data volume [GBytes]       " >> res
# echo "Memory read bandwidth [MBytes/s]  " >> res
# echo "Memory read data volume [GBytes]  " >> res
# echo "Memory write bandwidth [MBytes/s] " >> res
# echo "Memory write data volume [GBytes] " >> res
# echo "Memory bandwidth [MBytes/s]       " >> res
# echo "Memory data volume [GBytes]       " >> res
# for thread in {0..15}
# do
#   #echo ${thread} "threads profiling"
#   #likwid-perfctr -C S0:0\-${thread} -g MEM -m ./mem_profile bw
#   #likwid-perfctr -C S0:0\-${thread} -g MEM -m ./mem_profile bw | grep -E "bandwidth|volume" | sed 's/^|//' | sed 's/|$//' | sed 's/|/,/g'
#   printf "%10d\n" ${thread} > tmp
#   likwid-perfctr volume" | cut -d "|" -f 3 >> tmp
#   #sed -i '1i '${-C S0:0\-${thread} -g MEM -m ./mem_profile triad | grep -E "bandwidth|thread}'' tmp
#   paste -d "," res tmp > res_${thread}
#   mv res_${thread} res
#   cat res
# done

# rm res -rf
# echo "Memory read bandwidth [MBytes/s]  " >> res
# echo "Memory read data volume [GBytes]  " >> res
# echo "Memory write bandwidth [MBytes/s] " >> res
# echo "Memory write data volume [GBytes] " >> res
# echo "Memory bandwidth [MBytes/s]       " >> res
# echo "Memory data volume [GBytes]       " >> res
# #for log_index in {3..23..1}
# for kmax in {24576,40960,49152,57344,3145728,5242880,6291456,7340032}
# do
#   #let "kmax=2**${log_index}"
#   #echo $kmax
#   likwid-perfctr -C S0:0-0 -g MEM -m ./mem_profile 2d noblock 100 ${kmax} | grep -E "bandwidth|volume" | cut -d "|" -f 3 | tee tmp
#   paste -d "," res tmp > res_${kmax}
#   mv res_${kmax} res
# done
# awk -F, '{i=1;while(i <= NF){col[i]=col[i] $i ",";i=i+1}} END {i=1;while(i<=NF){print col[i];i=i+1}}' res | sed 's/[ ]*//g;s/,$//g' | tee 2d_noblock_mem_sup.csv

# rm res -rf
# echo "Memory read bandwidth [MBytes/s]  " >> res
# echo "Memory read data volume [GBytes]  " >> res
# echo "Memory write bandwidth [MBytes/s] " >> res
# echo "Memory write data volume [GBytes] " >> res
# echo "Memory bandwidth [MBytes/s]       " >> res
# echo "Memory data volume [GBytes]       " >> res
# for kmax in {24576,40960,49152,57344,3145728,5242880,6291456,7340032}
# #for log_index in {3..23..1}
# do
#   #let "kmax=2**${log_index}"
#   #echo $kmax
#   likwid-perfctr -C S0:0-0 -g MEM -m ./mem_profile 2d L2 100 ${kmax} | grep -E "bandwidth|volume" | cut -d "|" -f 3 | tee tmp
#   paste -d "," res tmp > res_${kmax}
#   mv res_${kmax} res
# done
# awk -F, '{i=1;while(i <= NF){col[i]=col[i] $i ",";i=i+1}} END {i=1;while(i<=NF){print col[i];i=i+1}}' res | sed 's/[ ]*//g;s/,$//g' | tee 2d_L2_mem_sup.csv

# rm res -rf
# echo "Memory read bandwidth [MBytes/s]  " >> res
# echo "Memory read data volume [GBytes]  " >> res
# echo "Memory write bandwidth [MBytes/s] " >> res
# echo "Memory write data volume [GBytes] " >> res
# echo "Memory bandwidth [MBytes/s]       " >> res
# echo "Memory data volume [GBytes]       " >> res
# #for log_index in {3..23..1}
# for kmax in {24576,40960,49152,57344,3145728,5242880,6291456,7340032}
# do
#   #let "kmax=2**${log_index}"
#   #echo $kmax
#   likwid-perfctr -C S0:0-0 -g MEM -m ./mem_profile 2d L3 100 ${kmax} | grep -E "bandwidth|volume" | cut -d "|" -f 3 | tee tmp
#   paste -d "," res tmp > res_${kmax}
#   mv res_${kmax} res
# done
# awk -F, '{i=1;while(i <= NF){col[i]=col[i] $i ",";i=i+1}} END {i=1;while(i<=NF){print col[i];i=i+1}}' res | sed 's/[ ]*//g;s/,$//g' | tee 2d_L3_mem_sup.csv

# rm res -rf
# for threads in {1,4,8,12,16}
# do
#   export OMP_NUM_THREADS=${threads}
#   let "t=${threads}-1"
#   #echo ${t}
#   likwid-perfctr -C S0:0\-${t} -g MEM -m ./mem_profile 3d noblock | grep -E "bandwidth|volume" | tee 3d_t${threads}
#   echo "Memory read bandwidth [MBytes/s],Memory read data volume [GBytes],Memory write bandwidth [MBytes/s],Memory write data volume [GBytes],Memory bandwidth [MBytes/s],Memory data volume [GBytes]" > output
#   lines=`wc -l 3d_t${threads} | cut -d " " -f 1`
#   echo ${lines}
#   for ((line=1; line<=${lines};line+=6))
#   do
#     let "begin=${line}"
#     let "end=${line}+5"
#     cut -d "|" -f 3 3d_t${threads} | sed  -n "${begin},${end}p" | awk '{i=1;while(i <= NF){col[i]=col[i] $i ",";i=i+1}} END {i=1;while(i<=NF){print col[i];i=i+1}}' >> output
#   done
#   mv output 3d_t${threads}.csv
# done

rm res -rf
for cache_size in {8.0,16.0,38.0,76.0,1000.0}
do
  export OMP_NUM_THREADS=16
  echo ${cache_size}
  likwid-perfctr -C S1:48-63 -g MEM -m ./mem_profile 3d block ${cache_size} | grep -E "bandwidth|volume" | tee ${cache_size}_mem_profile
  echo "Memory read bandwidth [MBytes/s],Memory read data volume [GBytes],Memory write bandwidth [MBytes/s],Memory write data volume [GBytes],Memory bandwidth [MBytes/s],Memory data volume [GBytes]" > output
  lines=`wc -l ${cache_size}_mem_profile | cut -d " " -f 1`
  echo ${lines}
  for ((line=1; line<=${lines};line+=6))
  do
    let "begin=${line}"
    let "end=${line}+5"
    cut -d "|" -f 3 ${cache_size}_mem_profile | sed  -n "${begin},${end}p" | awk '{i=1;while(i <= NF){col[i]=col[i] $i ",";i=i+1}} END {i=1;while(i<=NF){print col[i];i=i+1}}' >> output
  done
  mv output ${cache_size}_mem_profile.csv
done

# for file_item in {3d_block_mem,3d_block_mem}
# do
# echo "Memory read bandwidth [MBytes/s],Memory read data volume [GBytes],Memory write bandwidth [MBytes/s],Memory write data volume [GBytes],Memory bandwidth [MBytes/s],Memory data volume [GBytes]" > output
#   lines=`wc -l ${file_item} | cut -d " " -f 1`
#   echo ${lines}
#   for ((line=1; line<=${lines};line+=6))
#   do
#     let "begin=${line}"
#     let "end=${line}+5"
#     cut -d "|" -f 3 ${file_item} | sed  -n "${begin},${end}p" | awk '{i=1;while(i <= NF){col[i]=col[i] $i ",";i=i+1}} END {i=1;while(i<=NF){print col[i];i=i+1}}' >> output
#   done
#   mv output ${file_item}.csv
# done

### jloop parallel
# rm output -rf
# for thread in {1,4,8,12,16}
# do
#   export OMP_NUM_THREADS=${thread}
#   let "t=${thread}-1"
#   #echo ${t}
#   likwid-perfctr -C S0:0\-${t} -g MEM -m ./mem_profile 3d jparallel | grep -E "bandwidth|volume" | tee 3d_jparallel_t${thread}
#   echo "Memory read bandwidth [MBytes/s],Memory read data volume [GBytes],Memory write bandwidth [MBytes/s],Memory write data volume [GBytes],Memory bandwidth [MBytes/s],Memory data volume [GBytes]" > output
#   lines=`wc -l 3d_jparallel_t${thread} | cut -d " " -f 1`
#   echo ${lines}
#   for ((line=1; line<=${lines};line+=6))
#   do
#     let "begin=${line}"
#     let "end=${line}+5"
#     cut -d "|" -f 3 3d_jparallel_t${thread} | sed  -n "${begin},${end}p" | awk '{i=1;while(i <= NF){col[i]=col[i] $i ",";i=i+1}} END {i=1;while(i<=NF){print col[i];i=i+1}}' >> output
#   done
#   mv output 3d_jparallel_t${thread}.csv
# done

###jblock mem
# rm output -rf
# export OMP_NUM_THREADS=16
# likwid-perfctr -C S0:0\-${t} -g MEM -m ./mem_profile 3d jblock | grep -E "bandwidth|volume" | tee 3d_jblock_t16
# echo "Memory read bandwidth [MBytes/s],Memory read data volume [GBytes],Memory write bandwidth [MBytes/s],Memory write data volume [GBytes],Memory bandwidth [MBytes/s],Memory data volume [GBytes]" > output
# lines=`wc -l 3d_jblock_t16 | cut -d " " -f 1`
# echo ${lines}
# for ((line=1; line<=${lines};line+=6))
# do
#   let "begin=${line}"
#   let "end=${line}+5"
#   cut -d "|" -f 3 3d_jblock_t16 | sed  -n "${begin},${end}p" | awk '{i=1;while(i <= NF){col[i]=col[i] $i ",";i=i+1}} END {i=1;while(i<=NF){print col[i];i=i+1}}' >> output
# done
# mv output 3d_jblock_t16.csv

###jblock L3
# rm output -rf
# export OMP_NUM_THREADS=16
# likwid-perfctr -C S0:0-15 -g L3 -m ./mem_profile 3d jblock | grep -E "bandwidth|volume" | tee 3d_jblock_L3
#echo "L3 load bandwidth [MBytes/s] ,L3 load data volume [GBytes],L3 evict bandwidth [MBytes/s], L3 evict data volume [GBytes],L3|MEM evict bandwidth [MBytes/s],L3|MEM evict data volume [GBytes],Dropped CLs bandwidth [MBytes/s],Dropped CLs data volume [GBytes],L3 bandwidth [MBytes/s],L3 data volume [GBytes]" > output
#lines=`wc -l 3d_jblock_L3 | cut -d " " -f 1`
#echo ${lines}
#for ((line=1; line<=${lines};line+=10))
#do
#  let "begin=${line}"
#  let "end=${line}+9"
#  cut -d "|" -f 3 3d_jblock_L3 | sed  -n "${begin},${end}p" | awk '{i=1;while(i <= NF){col[i]=col[i] $i ",";i=i+1}} END {i=1;while(i<=NF){print col[i];i=i+1}}' >> output
#done
#mv output 3d_jblock_L3.csv