build:
	mpicxx -fopenmp -c main.c 
	mpicxx -fopenmp -c funcs.c 
	nvcc -I./inc -c cudaFunctions.cu 
	mpicxx -fopenmp -o exec main.o funcs.o cudaFunctions.o /usr/local/cuda-11.0/lib64/libcudart_static.a -ldl -lrt
	# mpicc -o exec main.c funcs.c

clean:
	rm -f *.o ./exec

run:
	mpiexec -n 3 ./exec < input.txt
