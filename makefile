build:
	mpicc -fopenmp -c main.c -o main.o
	mpicc -fopenmp -c funcs.c -o funcs.o
	# nvcc -I./inc -c cudaFunctions.cu -o cudaFunctions.o
	# mpicxx -fopenmp -o exec main.o funcs.o cudaFunctions.o /usr/local/cuda-11.0/lib64/libcudart_static.a -ldl -lrt
	mpicc -fopenmp -o exec main.o funcs.o
	mpicc -o exec main.c funcs.c

clean:
	rm -f *.o ./exec

run:
	mpiexec -n 3 ./exec < input.txt
