CXX         := g++
CXXFLAGS    := -O3 -Wall -I ./include
LDFLAGS     := -lm


SRC = $(wildcard ./src/*.cpp)  
KERNELS = $(wildcard ./src/kernels/*.cpp)
OBJ = $(patsubst %.cpp, %.o, $(SRC))
KERNEL_OBJS = $(patsubst %.cpp, %.o, $(KERNELS))


pe:$(OBJ) $(KERNEL_OBJS)
	$(CXX) $(OBJ) $(KERNEL_OBJS) -o $@

./src/%.o: ./src/%.cpp
	$(CXX) $(CFLAGS) -c -g $< -o $@

./src/kernel/%.o: ./src/kernels/%.cpp
	$(CXX) $(CFLAGS) -c  -g $< -o $@


all: 
	#@echo $(SRC)
	#@echo $(OBJ)
	#@echo $(KERNEL_OBJS)
	pe

clean:
	rm ./src/*.o
	rm ./src/kernels/*.o
	rm pe
