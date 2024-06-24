#!/bin/bash
#SBATCH --job-name=random2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

source ./env.sh

./build.sh

export LATENCY_OUTPUT_FILENAME_PREFIX="mem_random_without_ptrchase_thread2"
export OMP_NUM_THREADS=2

numactl -N 0 -m 0 ./bin/mem_random_without_ptrchase_omp
