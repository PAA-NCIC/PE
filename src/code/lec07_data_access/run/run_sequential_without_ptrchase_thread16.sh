#!/bin/bash
#SBATCH --partition=a800
#SBATCH --job-name=seqential16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=g07
#SBATCH --exclusive

source ./env.sh

./build.sh


export LATENCY_OUTPUT_FILENAME_PREFIX="cycle_seqential_without_ptrchase_thread16"
export OMP_NUM_THREADS=16

numactl -N 0 -m 0 ./bin/mem_seqential_without_ptrchase_omp
