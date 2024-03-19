#!/bin/bash
#SBATCH --job-name=dmvm_mflops_avx512_unroll4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

source ./env.sh

./build.sh

export LATENCY_OUTPUT_FILENAME_PREFIX="dmvm_mflops_avx512_unroll4"

numactl -N 0 -m 0 ./bin/dmvm_avx512_unroll4
