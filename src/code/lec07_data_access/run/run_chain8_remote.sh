#!/bin/bash
#SBATCH --job-name=remote_chain8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

source ./env.sh

./build.sh

export SAMPLE_POINTS=4
# export ACCESS_REGION_START=256
# export 23615=268435456
export LATENCY_OUTPUT_FILENAME_PREFIX="mem_08_remote"

numactl -N 0 -m 1 ./bin/mem_chain8