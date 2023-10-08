#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=chain8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=g02
#SBATCH --exclusive

source ./env.sh

./build.sh

export SAMPLE_POINTS=4
# export ACCESS_REGION_START=256
# export 23615=268435456
export LATENCY_OUTPUT_FILENAME_PREFIX="cycle_08"

numactl -N 0 -m 0 ./bin/mem_chain8
