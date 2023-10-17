#!/bin/bash
#SBATCH --partition=a800
#SBATCH --job-name=fused
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=g07
#SBATCH --exclusive

source ./env.sh

./build.sh

export LATENCY_OUTPUT_FILENAME_NAME="fused"

numactl -N 0 -m 0 ./bin/two_scale_fused
