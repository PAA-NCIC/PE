CXX         := g++
CXXFLAGS    := -O3 -march=native -D __AVX512F__
LDFLAGS     := -lm


SRC = $(wildcard ./*.cpp)  
KERNELS = $(wildcard ./src/kernels/*.cpp)
OBJ = $(patsubst %.cpp, %.o, $(SRC))
KERNEL_OBJS = $(patsubst %.cpp, %.o, $(KERNELS))

pe:example1 example2

example1: example1.cpp
	$(CXX) $(CXXFLAGS)  $< -o $@

example2: example2.cpp
	$(CXX) $(CXXFLAGS)  $< -o $@

all: 
	#@echo $(SRC)
	#@echo $(OBJ)
	#@echo $(KERNEL_OBJS)
	pe

clean:
	rm pe
