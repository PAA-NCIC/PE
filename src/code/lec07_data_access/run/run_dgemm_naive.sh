#!/bin/bash
#SBATCH --job-name=dgemm_naive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

source ./env.sh

./build.sh

export LATENCY_OUTPUT_FILENAME_NAME="naive"

numactl -N 0 -m 0 ./bin/dgemm_naive
