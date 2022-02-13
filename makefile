build:
	mpicxx -fopenmp -c main.c -o main.o
	mpicxx -fopenmp -c funcs.c -o funcs.o
	nvcc -I./inc -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -o exec main.o funcs.o cudaFunctions.o /usr/local/cuda-11.0/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./exec