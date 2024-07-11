OLD  := MMult00
NEW	 := MMult00
#NEW  := MMult01_register_cache_c
#NEW  := MMult02_8x1_unroll_row
#NEW	 := MMult03_8x1_simd
#NEW	 := MMult04_8x8_tiling
#NEW	 := MMult05_24x8_tiling
#NEW	 := MMult06_24x8_blocking
#NEW	 := MMult07_tt24x8_b240x256
#NEW	 := MMult08_tt24x8_b240x256_pA
#NEW	 := MMult09_tt24x8_b240x256_pAB
#NEW	 := MMult10_prefetchA
#NEW	 := MMult11_prefetchAB
#NEW	 := MMult12_prefetchABC
#NEW	 := MMult13_finetune
#NEW	 := MMult_mkl

#
# sample makefile
#

CC         := gcc
LINKER     := $(CC)
CFLAGS     := -O2 -Wall -march=native -Wunused-function
MKLPATH		= /opt/intel/oneapi/mkl/2024.0
MKLPATH		= /opt/intel/oneapi/mkl/2024.0
LDFLAGS    := -lm -L$(MKLPATH)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -DMKL_ILP64 -m64
INCFLAGS	= -I$(MKLPATH)/include


UTIL       := copy_matrix.o \
              compare_matrices.o \
              random_matrix.o \
              dclock.o \
              REF_MMult.o \
              print_matrix.o

TEST_OBJS  := test_MMult.o $(NEW).o 

%.o: %.c
	$(CC)  $(INCFLAGS) $(CFLAGS) -c $< -o $@

all: 
	make clean;
	make test_MMult.x

test_MMult.x: $(TEST_OBJS) $(UTIL) parameters.h
	$(LINKER) $(TEST_OBJS) $(UTIL) $(LDFLAGS) \
        $(BLAS_LIB) -o $(TEST_BIN) $@ 

run:	
	make all
	export OMP_NUM_THREADS=1
	export MKL_NUM_THREADS=1
	echo "version = '$(NEW)';" > output_$(NEW).m
	./test_MMult.x | tee -a output_$(NEW).m
	cp output_$(OLD).m output_old.m
	cp output_$(NEW).m output_new.m

clean:
	rm -f *.o *~ core *.x

asm:
	$(CC)  $(INCFLAGS) $(CFLAGS) -S $(NEW).c 

exe:
	$(CC)  $(INCFLAGS) $(CFLAGS) -c $(NEW).s -o $(NEW).o
	make test_MMult.x

cleanall:
	rm -f *.o *~ core *.x output*.m *.eps *.png
