#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=local
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=g03
#SBATCH --exclusive

source ./env.sh

./build.sh

# export SAMPLE_POINTS=16
# export ACCESS_REGION_START=256

export LATENCY_OUTPUT_FILENAME_PREFIX="cycle_local"

numactl -N 0 -m 0 ./bin/mem
