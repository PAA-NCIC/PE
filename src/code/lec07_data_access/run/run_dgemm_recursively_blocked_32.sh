#!/bin/bash
#SBATCH --partition=a800
#SBATCH --job-name=b32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=g07
#SBATCH --exclusive

source ./env.sh

./build.sh

export LATENCY_OUTPUT_FILENAME_NAME="recursively_blocked_32"
export BLOCK_SIZE=32

numactl -N 0 -m 0 ./bin/dgemm_recursively_blocked
