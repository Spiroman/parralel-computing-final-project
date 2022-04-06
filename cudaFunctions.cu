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

__device__ char conservativeGroup[GROUP_A_ROWS][GROUP_A_COLS] = {
    "NDEQ",
    "MILV",
    "FYW",
    "NEQK",
    "QHRK",
    "HY",
    "STA",
    "NHQK",
    "MILF"};

__device__ char semiConservativeGroup[GROUP_B_ROWS][GROUP_B_COLS] = {
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

__device__ int checkConservativeGroup(char seq1, char seq2)
{

    for (int i = 0; i < GROUP_A_ROWS; i++)
    {
        for (int j = 0; j < GROUP_A_COLS; j++)
        {
            if ((semiConservativeGroup[i][j] == seq1) && (semiConservativeGroup[i][j] == seq2))
                return 1;
        }
    }
    return 0;
}

__device__ int checkSemiConservativeGroup(char seq1, char seq2)
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


__global__ void determinePartialScores(char *baseSeq, char *mutation, int *cmpRes, int* weights){
    int tx = threadIdx.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= num_of_checks)
        return;
    if (tid < num_of_checks)
    {
        // For each type of match/missmatch we will assign the score of the match directly instead of the the char.
        // Meaning: if we have a full match, we will assign the weight of the full match in the result array
        // instead of putting '*', which will make the final calculation quicker (less checks for the type of char in each index)
        if (baseSeq[tid] == mutation[tid])
        {
            // Complete match -> '*' in our assignment
            cmpRes[tid] = weights[0];
        }
        else if (checkConservativeGroup(baseSeq[tid],mutation[tid]))
        {
            // Conservative match -> ':' in our assignment
            cmpRes[tid] = weights[1];
        }
        else if (checkSemiConservativeGroup(baseSeq[tid], mutation[tid]))
        {
            // Semi-Conservative match -> '.' in our assignment
            cmpRes[tid] = weights[2];
        }
        else
        {
            // Not a match -> ' ' in our assignment
            cmpRes[tid] = weights[3];
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

void launch_cuda(char *baseSeq, char *mutation, int lenOfAugmented, int *cmpRes, int *weights)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate memory on GPU
    err = cudaMalloc((void **)&cuda_baseSeq, lenOfAugmented);
    checkErr(err, "Failed to allocate device memory seq2");

    err = cudaMalloc((void **)&cuda_mutation, lenOfAugmented);
    checkErr(err, "Failed to allocate device memory seq1");

    err = cudaMalloc((void **)&cuda_cmpRes, lenOfAugmented);
    checkErr(err, "Failed to allocate device memory w_cuda-");

    err = cudaMalloc((void **)&cuda_weights, WEIGHTS);
    checkErr(err, "Failed to allocate device memory w_cuda-");

    // Copy from host to device
    err = cudaMemcpy(cuda_baseSeq, baseSeq, lenOfAugmented, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy data from host to device seq2 -");

    err = cudaMemcpy(cuda_mutation, mutation, lenOfAugmented, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy data from host to device seq1 -");

    err = cudaMemcpy(cuda_weights, weights, WEIGHTS, cudaMemcpyHostToDevice);
    checkErr(err, "Failed to copy data from host to device w_cuda -");

    // Calculate the number of blocks
    int blocksPerGrid = (lenOfAugmented + MAX_THREADS - 1) / MAX_THREADS;

    // Launch the Kernel
    determinePartialScores<<<blocksPerGrid, MAX_THREADS>>>(char *cuda_baseSeq, char *cuda_mutation, int *cuda_cmpRes, int *cuda_weights);

    err = cudaDeviceSynchronize();
    checkErr(err, "Failed to synch kernel -");

    err = cudaGetLastError();
    checkErr(err, "Failed kernel -");
    
    // Copy results
    err = cudaMemcpy(cmpRes, cuda_cmpRes, lenOfAugmented, cudaMemcpyDeviceToHost);
    checkErr(err, "Failed to copy data device to host results -");

    err = cudaFree(cuda_baseSeq);
    checkErr(err, "Failed to free seq1 -");

    err = cudaFree(cuda_mutation);
    checkErr(err, "Failed to free seq2 -");

    err = cudaFree(cuda_cmpRes);
    checkErr(err, "Failed to free results -");   

    err = cudaFree(cuda_weights);
    checkErr(err, "Failed to free weights -");
    
    return;
}


