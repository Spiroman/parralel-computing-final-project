#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <stddef.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>
#include "data.h"
#include "cudaFunctions.h"

#define MAX_THREADS 256
#define WEIGHTS 4

__device__ char group_one[GROUP_A_ROWS][GROUP_A_COLS] = {
    "NDEQ",
    "MILV",
    "FYW",
    "NEQK",
    "QHRK",
    "HY",
    "STA",
    "NHQK",
    "MILF"};

__device__ char group_two[GROUP_B_ROWS][GROUP_B_COLS] = {
    "SAG",
    "SGND",
    "NEQHRK",
    "HFY",
    "ATV",
    "STPA",
    "NDEQHK",
    "FVLIM",
    "CSA",
    "STNK",
    "SNDEQK"};

__device__ int compare_group_a_cuda(char seq1, char seq2)
{

    for (int i = 0; i < GROUP_A_ROWS; i++)
    {
        for (int j = 0; j < GROUP_A_COLS; j++)
        {
            if ((group_two[i][j] == seq1) && (group_two[i][j] == seq2))
                return 1;
        }
    }
    return 0;
}

__device__ int compare_group_b_cuda(char seq1, char seq2)
{
    for (int i = 0; i < GROUP_A_ROWS; i++)
    {
        for (int j = 0; j < GROUP_A_COLS; j++)
        {
                if ((group_two[i][j]== seq1) && (group_two[i][j] == seq2))
                    return 1;
        }
    }
    return 0;
}

__device__ char compare_cuda(char seq1, char seq2)
{

    if (seq1 == seq2) 
    {
        return '$';
    }
    else if (compare_group_a_cuda(seq1, seq2) == 1)
    {
        return '%';
    }
    else if (compare_group_b_cuda(seq1, seq2) == 1)
    {
        return '#';
    }
    else
    {
        return ' ';
    }
}

__device__ int calc_score_cuda(char *seq1, char *seq2, int seq1_len, int seq2_len, int* w)
{
    int weights[WEIGHTS] = {0};
    int alignment_score = 0;
    char compare_res;

    for (int i = 0; i < seq2_len; i++)
    {
        compare_res = compare_cuda(seq1[i], seq2[i]);
        if (compare_res == '$')
            weights[0]++;
        else if (compare_res == '%')
            weights[1]++;
        else if (compare_res == '#')
            weights[2]++;
        else
            weights[3]++;
    }
    alignment_score = (w[0] * weights[0]) - (w[1] * weights[1]) - (w[2] * weights[2]) - (w[3] * weights[3]);

    return alignment_score;
}

__global__ void find_best_score(char *seq1, char *seq2, int seq1_len, int seq2_len, int w[4], int num_of_checks, Result* results)
{
    int tx = threadIdx.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int temp_score_cuda =0;

    if(tid >= num_of_checks)
        return;
    if (tid < num_of_checks)
    {
        temp_score_cuda = calc_score_cuda(seq1+tid, seq2, seq1_len - tid, seq2_len, w);

        if (temp_score_cuda > results->score || results->score == 0)
            {
                results->n = tid;
                results->score = temp_score_cuda;
            }
    }
}

void checkErr(cudaError_t err, const char* s_err)
{
      if (err != cudaSuccess)
    {
        fprintf(stderr, "%s - %s\n", s_err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int launch_cuda(char* seq1, int seq1_len, char* seq2, int seq2_len,int num_of_checks, int w[4], Result* results)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int size_seq1 = seq1_len * sizeof(char) +1;
    int size_seq2 = seq2_len * sizeof(char) +1;
    int sizeWeight= 4 * sizeof(int);

    char *seq2_cuda;
    char *seq1_cuda;
    int *w_cuda;
    int temp_score_cuda = 0;

    Result* cuda_results;

    // Allocate memory on GPU to copy seq1 from the host
    err = cudaMalloc((void **)&cuda_results, sizeof(Result));
    checkErr(err, "Failed to allocate device memory results");

    err = cudaMalloc((void **)&seq2_cuda, size_seq2);
    checkErr(err, "Failed to allocate device memory seq2");

    err = cudaMalloc((void **)&seq1_cuda, size_seq1);
    checkErr(err, "Failed to allocate device memory seq1");

    err = cudaMalloc((void **)&w_cuda, sizeWeight);
    checkErr(err, "Failed to allocate device memory w_cuda-");

    err = cudaMemcpy(seq2_cuda, seq2, size_seq2, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy data from host to device seq2 -");

    err = cudaMemcpy(seq1_cuda, seq1, size_seq1, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy data from host to device seq1 -");

    err = cudaMemcpy(w_cuda, w, sizeWeight, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy data from host to device w_cuda -");

    err = cudaMemcpy(cuda_results, results, sizeof(Result), cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy data from host to device results -");
    
    int blocksPerGrid = (num_of_checks + MAX_THREADS - 1) / MAX_THREADS;
    // Launch the Kernel
    find_best_score<<<blocksPerGrid, MAX_THREADS>>>(seq1_cuda, seq2_cuda ,seq1_len, seq2_len, w_cuda, num_of_checks, cuda_results);
    
    err = cudaDeviceSynchronize();
    checkErr(err, "Failed to synch kernel -");

    err = cudaGetLastError();
    checkErr(err, "Failed kernel -");

    err = cudaMemcpy(results, cuda_results, sizeof(Result), cudaMemcpyDeviceToHost);
    checkErr(err, "Failed to copy data device to host results -");

    err = cudaFree(seq1_cuda);
    checkErr(err, "Failed to free seq1 -");

    err = cudaFree(seq2_cuda);
    checkErr(err, "Failed to free seq2 -");

    err = cudaFree(w_cuda);
    checkErr(err, "Failed to free weights -");

    err = cudaFree(cuda_results);
    checkErr(err, "Failed to free results -");   

    return temp_score_cuda;
}


