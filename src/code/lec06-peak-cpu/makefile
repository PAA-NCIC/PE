CXX         := g++
CXXFLAGS    := -O3 -march=native -D __AVX512F__ -g
LDFLAGS     := -lm


SRC = $(wildcard ./*.cpp)  
KERNELS = $(wildcard ./src/kernels/*.cpp)
OBJ = $(patsubst %.cpp, %.o, $(SRC))
KERNEL_OBJS = $(patsubst %.cpp, %.o, $(KERNELS))

pe:example1 example1-unroll example2 example2-multi-chains example3 example4 example5 example6

example1: example1.cpp
	$(CXX) $(CXXFLAGS)  $< -o $@

example1-unroll: example1-unroll.cpp
	$(CXX) $(CXXFLAGS)  $< -o $@

example2: example2.cpp
	$(CXX) $(CXXFLAGS)  $< -o $@

example2-multi-chains: example2-multi-chains.cpp
	$(CXX) $(CXXFLAGS)  $< -o $@

example3: example3.cpp
	$(CXX) $(CXXFLAGS)  $< -o $@

example4: example4.cpp
	$(CXX) $(CXXFLAGS)  $< -o $@
	
example5: example5.cpp
	$(CXX) $(CXXFLAGS)  $< -o $@

example6: example6.cpp
	$(CXX) $(CXXFLAGS)  $< -o $@

all: 
	#@echo $(SRC)
	#@echo $(OBJ)
	#@echo $(KERNEL_OBJS)
	pe

clean:
	rm example1 example1-unroll example2  example3 example4 example5  example6 *.s -rf
