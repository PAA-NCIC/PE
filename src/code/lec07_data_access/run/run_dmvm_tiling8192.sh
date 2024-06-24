#!/bin/bash
#SBATCH --job-name=dmvm_mflops_tiling8192
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

source ./env.sh

./build.sh

export LATENCY_OUTPUT_FILENAME_PREFIX="dmvm_mflops_tiling8192"
export ROW_TILING=8192

numactl -N 0 -m 0 ./bin/dmvm_tiling
