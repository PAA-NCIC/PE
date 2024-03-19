#!/bin/bash
#SBATCH --job-name=remote_chain6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

source ./env.sh

./build.sh

# export SAMPLE_POINTS=16
# export ACCESS_REGION_START=256
# export ACCESS_REGION_END=268435456
export LATENCY_OUTPUT_FILENAME_PREFIX="mem_06_remote"

numactl -N 0 -m 1 ./bin/mem_chain6
