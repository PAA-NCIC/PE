#!/bin/bash

source ./env.sh

CC="icc"
CFLAGS="-O2 -std=c11"
# CFLAGS="-O1 -std=c11 -march=cascadelake"

set -ex

$CC $CFLAGS -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=1  -DDEF_GEN_RANDOM_LIST mem.c -o ./bin/mem
$CC $CFLAGS -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=2  -DDEF_GEN_RANDOM_LIST mem.c -o ./bin/mem_chain2
$CC $CFLAGS -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=4  -DDEF_GEN_RANDOM_LIST mem.c -o ./bin/mem_chain4
$CC $CFLAGS -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=6  -DDEF_GEN_RANDOM_LIST mem.c -o ./bin/mem_chain6
$CC $CFLAGS -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=8  -DDEF_GEN_RANDOM_LIST mem.c -o ./bin/mem_chain8
$CC $CFLAGS -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=10 -DDEF_GEN_RANDOM_LIST mem.c -o ./bin/mem_chain10
$CC $CFLAGS -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=12 -DDEF_GEN_RANDOM_LIST mem.c -o ./bin/mem_chain12
$CC $CFLAGS -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=14 -DDEF_GEN_RANDOM_LIST mem.c -o ./bin/mem_chain14
$CC $CFLAGS -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=16 -DDEF_GEN_RANDOM_LIST mem.c -o ./bin/mem_chain16


$CC $CFLAGS -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=1 -DDEF_GEN_SEQUENTIAL_LIST mem.c -o ./bin/mem_address_ordered_traveral

$CC $CFLAGS -DDEF_CHAIN_COUNT=1 -DDEF_GEN_RANDOM_LIST -DDEF_RANDOM_WITHOUT_PTRCHASE mem.c -o ./bin/mem_random_without_ptrchase
$CC $CFLAGS -DDEF_CHAIN_COUNT=1 -DDEF_GEN_SEQUENTIAL_LIST -DDEF_SEQENTIAL_WITHOUT_PTRCHASE mem.c -o ./bin/mem_seqential_without_ptrchase

$CC $CFLAGS -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=1  -DDEF_GEN_RANDOM_LIST -DDEF_PREFETCH mem.c -o ./bin/mem_prefetch


$CC $CFLAGS -fopenmp -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=1  -DDEF_GEN_RANDOM_LIST mem.c -o ./bin/mem_chain1_omp
$CC $CFLAGS -fopenmp -DDEF_PTRCHASE -DDEF_CHAIN_COUNT=10  -DDEF_GEN_RANDOM_LIST mem.c -o ./bin/mem_chain10_omp

$CC $CFLAGS -fopenmp -DDEF_CHAIN_COUNT=1 -DDEF_GEN_RANDOM_LIST -DDEF_RANDOM_WITHOUT_PTRCHASE mem.c -o ./bin/mem_random_without_ptrchase_omp
$CC $CFLAGS -fopenmp -DDEF_CHAIN_COUNT=1 -DDEF_GEN_SEQUENTIAL_LIST -DDEF_SEQENTIAL_WITHOUT_PTRCHASE mem.c -o ./bin/mem_seqential_without_ptrchase_omp

$CC $CFLAGS -DVECTOR_TRIADS_NAIVE triads.c -o ./bin/triads_naive
$CC $CFLAGS -DVECTOR_TRIADS_AVX512 triads.c -o ./bin/triads_avx512
$CC $CFLAGS -DVECTOR_TRIADS_AVX512_NT triads.c -o ./bin/triads_avx512_nt
$CC $CFLAGS -DVECTOR_TRIADS_MODEL triads.c -o ./bin/triads_model

$CC $CFLAGS -DDMVM_NAIVE dmvm.c -o ./bin/dmvm_naive
$CC $CFLAGS -DDMVM_UNROLL2 dmvm.c -o ./bin/dmvm_unroll2
$CC $CFLAGS -DDMVM_TILING dmvm.c -o ./bin/dmvm_tiling
$CC $CFLAGS -DDMVM_DATA_TRAFFIC_NAIVE dmvm.c -o ./bin/dmvm_data_traffic_naive
$CC $CFLAGS -DDMVM_DATA_TRAFFIC_UNROLL2 dmvm.c -o ./bin/dmvm_data_traffic_unroll2
$CC $CFLAGS -DDMVM_DATA_TRAFFIC_TILING dmvm.c -o ./bin/dmvm_data_traffic_tiling
