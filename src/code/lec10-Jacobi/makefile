CXX         := g++
CXXFLAGS    := -O3 -Wall -I ./include -fopenmp -g
LDFLAGS     := -lm


SRC = $(wildcard ./src/*.cpp)  
KERNELS = $(wildcard ./src/kernels/*.cpp)
OBJ = $(patsubst %.cpp, %.o, $(SRC))
KERNEL_OBJS = $(patsubst %.cpp, %.o, $(KERNELS))


pe:$(OBJ) $(KERNEL_OBJS)
	$(CXX) $(OBJ) $(KERNEL_OBJS) -o $@ -fopenmp

./src/%.o: ./src/%.cpp
	$(CXX) $(CXXFLAGS) -c  $< -o $@ 

./src/kernel/%.o: ./src/kernels/%.cpp
	$(CXX) $(CXXFLAGS) -c  $< -o $@ 


all: 
	#@echo $(SRC)
	#@echo $(OBJ)
	#@echo $(KERNEL_OBJS)
	pe

clean:
	rm ./src/*.o
	rm ./src/kernels/*.o
	rm pe
