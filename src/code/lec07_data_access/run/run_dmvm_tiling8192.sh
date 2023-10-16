#!/bin/bash
#SBATCH --partition=a800
#SBATCH --job-name=tiling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=g07
#SBATCH --exclusive

source ./env.sh

./build.sh

export LATENCY_OUTPUT_FILENAME_PREFIX="dmvm_mflops_tiling8192"
export ROW_TILING=8192

numactl -N 0 -m 0 ./bin/dmvm_tiling
