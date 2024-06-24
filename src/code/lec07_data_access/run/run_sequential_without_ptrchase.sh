#!/bin/bash
#SBATCH --job-name=sequential
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

source ./env.sh

./build.sh


export LATENCY_OUTPUT_FILENAME_PREFIX="mem_seqential_without_ptrchase"

numactl -N 0 -m 0 ./bin/mem_seqential_without_ptrchase
