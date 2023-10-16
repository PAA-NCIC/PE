#!/bin/bash
#SBATCH --partition=a800
#SBATCH --job-name=naive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=g07
#SBATCH --exclusive

source ./env.sh

./build.sh

export LATENCY_OUTPUT_FILENAME_PREFIX="dmvm_data_traffic_tiling2048"
export ROW_TILING=2048

numactl -N 0 -m 0 ./bin/dmvm_data_traffic_tiling
