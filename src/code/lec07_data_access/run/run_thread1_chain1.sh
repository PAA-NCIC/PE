#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=t1c1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=g01
#SBATCH --exclusive

source ./env.sh

./build.sh

export LATENCY_OUTPUT_FILENAME_PREFIX="cycle_thread1_chain1"
export OMP_NUM_THREADS=1

numactl -N 0 -m 0 ./bin/mem_chain1_omp
