#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=chain14
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=g01
#SBATCH --exclusive

source ./env.sh

./build.sh

export SAMPLE_POINTS=4
# export ACCESS_REGION_START=256
# export ACCESS_REGION_END=268435456
export LATENCY_OUTPUT_FILENAME_PREFIX="cycle_14"

numactl -N 0 -m 0 ./bin/mem_chain14
