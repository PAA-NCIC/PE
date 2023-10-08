#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=seqential8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=g03
#SBATCH --exclusive

source ./env.sh

./build.sh


export LATENCY_OUTPUT_FILENAME_PREFIX="cycle_seqential_without_ptrchase_thread8"
export OMP_NUM_THREADS=8

numactl -N 0 -m 0 ./bin/mem_seqential_without_ptrchase_omp
