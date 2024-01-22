#!/bin/bash
#print table head

# 3d iparallel noblock
#for threads in {1,4,8,12,16}
#do
#  export OMP_NUM_THREADS=${threads}
#  let "t=${threads}-1"
#  likwid-pin -c S0:0\-${t} ./pe 3d noblock
#done

# 3d iparallel block, test cache size in MB
# for cache_size in {8.0,16.0,38.0,76.0,1000.0}
# do
#   export OMP_NUM_THREADS=16
#   likwid-pin -c S0:0-15 ./pe 3d block ${cache_size} | tee ${cache_size}.perf
# done

#3d jblock size test
# export OMP_NUM_THREADS=16
# likwid-pin -c S1:48-63 ./pe 3d jblock | tee jblock.perf

#3d jparallel
# export OMP_NUM_THREADS=16
# likwid-pin -c S1:48-63 ./pe 3d jparallel | tee jparallel.perf

#3d ntstore
# export OMP_NUM_THREADS=16
# likwid-pin -c S1:48-63 ./pe 3d ntstore | tee ntstore.perf