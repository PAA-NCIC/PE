#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=random_without_ptrchase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=g03
#SBATCH --exclusive

# spack load numactl@2.0.14

./build.sh

export LATENCY_OUTPUT_FILENAME_PREFIX="cycle_random_without_ptrchase"
export LATENCY_OUTPUT_FILENAME_SUFFIX=".dat"

numactl -N 0 -m 0 ./bin/mem_random_without_ptrchase
