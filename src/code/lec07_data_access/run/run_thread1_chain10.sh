#!/bin/bash
#SBATCH --partition=a800
#SBATCH --job-name=t1c10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=g07
#SBATCH --exclusive

source ./env.sh

./build.sh

export LATENCY_OUTPUT_FILENAME_PREFIX="mem_thread1_chain10"
export OMP_NUM_THREADS=1

numactl -N 0 -m 0 ./bin/mem_chain10_omp
