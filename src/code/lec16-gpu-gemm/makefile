BINARY_NAME = bench_dgemm
CUDA_PATH   = /usr/local/cuda-11.7
CC			= $(CUDA_PATH)/bin/nvcc
CFLAGS		= -O3 -std=c++11
LDFLAGS		= -L/usr/local/cuda-11.7/targets/x86_64-linux/lib -lcudart -lcublas
INCFLAGS	= -I$(CUDA_PATH)/include  -I./include/common/inc


SRC = $(wildcard ./*.cu)  
KERNEL_TEMPLATES = $(wildcard ./include/*.cuh)
KERNELS = $(wildcard ./kernels/*.cu)
DEPENS = $(SRC) $(KERNELS) $(KERNEL_TEMPLATES)


build : $(BINARY_NAME)

$(BINARY_NAME): $(DEPENS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCFLAGS) $(KERNELS) $(SRC) -o $(BINARY_NAME)



clean:
	rm $(BINARY_NAME)
