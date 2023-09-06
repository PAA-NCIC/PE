#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=seqential_without_ptrchase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=g04
#SBATCH --exclusive

# spack load numactl@2.0.14

./build.sh

export ACCESS_REGION_START=256
export ACCESS_REGION_END=2147483648
export REPEAT_COUNT=1000

export LATENCY_OUTPUT_FILENAME_PREFIX="cycle_seqential_without_ptrchase"
export LATENCY_OUTPUT_FILENAME_SUFFIX=".dat"

numactl -N 0 -m 0 ./bin/mem_seqential_without_ptrchase
