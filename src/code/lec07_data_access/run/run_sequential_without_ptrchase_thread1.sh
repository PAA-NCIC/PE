#!/bin/bash
#SBATCH --job-name=sequential1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

source ./env.sh

./build.sh


export LATENCY_OUTPUT_FILENAME_PREFIX="mem_seqential_without_ptrchase_thread1"
export OMP_NUM_THREADS=1

numactl -N 0 -m 0 ./bin/mem_seqential_without_ptrchase_omp
