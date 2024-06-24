#!/bin/bash
#SBATCH --job-name=dmvm_mflops_tiling2048
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

source ./env.sh

./build.sh

export LATENCY_OUTPUT_FILENAME_PREFIX="dmvm_mflops_tiling2048"
export ROW_TILING=2048

numactl -N 0 -m 0 ./bin/dmvm_tiling
