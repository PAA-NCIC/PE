BINARY_NAME = pe_dgemm_gpu
CUDA_PATH   = /usr/local/cuda-11.7
CC			= $(CUDA_PATH)/bin/nvcc
CFLAGS		= -O3 -std=c++11
LDFLAGS		= -L/usr/local/cuda-11.7/targets/x86_64-linux/lib -lcudart -lcublas
INCFLAGS	= -I$(CUDA_PATH)/include  -I./include/common/inc


SRC = $(wildcard ./*.cu)  
KERNELS = $(wildcard ./kernels/*.cu)

OBJ = $(patsubst %.cu, %.o, $(SRC))
KERNEL_OBJS = $(patsubst %.cu, %.o, $(KERNELS))

build : $(BINARY_NAME)

$(BINARY_NAME): $(SRC) $(KERNELS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCFLAGS) $(KERNELS) $(SRC) -o $(BINARY_NAME)



clean:
	rm $(BINARY_NAME)
