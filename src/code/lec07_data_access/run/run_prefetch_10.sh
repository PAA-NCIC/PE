#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=prefetch10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=g02
#SBATCH --exclusive

source ./env.sh

./build.sh

# export SAMPLE_POINTS=16
# export ACCESS_REGION_START=256
# export ACCESS_REGION_END=268435456

export PREFETCH_COUNT=10
export LATENCY_OUTPUT_FILENAME_PREFIX="cycle_prefetch_10"

numactl -N 0 -m 0 ./bin/mem_prefetch
