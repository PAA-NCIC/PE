
all: hello axpy axpy.atx

hello: hello.cu
	nvcc hello.cu -o hello
axpy: axpy.cu
	nvcc -o axpy -O3 -Xptxas -O3 -x cu --generate-code arch=compute_70,code=sm_70 --compiler-options=-mavx2 axpy.cu
axpy.atx: axpy.cu
	nvcc -ptx -O3 -arch=compute_70 axpy.cu  -o axpy.ptx


clean:
	rm hello axpy axpy.ptx -rf

#nvcc -o axpy.nvcc -O3 -Xptxas -O3 -x cu --generate-code arch=compute 70,code=sm 70 --compiler-options=-mavx2 axpy.cc