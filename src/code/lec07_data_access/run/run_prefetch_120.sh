#!/bin/bash
#SBATCH --job-name=prefetch120
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

source ./env.sh

./build.sh

# export SAMPLE_POINTS=16
# export ACCESS_REGION_START=256
# export ACCESS_REGION_END=268435456

export PREFETCH_COUNT=120
export LATENCY_OUTPUT_FILENAME_PREFIX="mem_prefetch_120"

numactl -N 0 -m 0 ./bin/mem_prefetch
