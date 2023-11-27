#!/bin/bash

source env.sh

CC=gcc
CFLAGS="-O2 -g -std=c11"

rm -f ./bin/mem

$CC $CFLAGS mem.c -o ./bin/mem
