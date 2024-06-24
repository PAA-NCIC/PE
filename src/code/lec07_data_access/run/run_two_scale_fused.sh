#!/bin/bash
#SBATCH --job-name=two_scale_fused
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

source ./env.sh

./build.sh

export LATENCY_OUTPUT_FILENAME_NAME="fused"

numactl -N 0 -m 0 ./bin/two_scale_fused
