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
// #define DEBUG 1
// #define DEBUG_FINAL 1

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
            if ((conservativeGroup[i][j] == seq1) && (conservativeGroup[i][j] == seq2))
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
            if ((semiConservativeGroup[i][j] == seq1) && (semiConservativeGroup[i][j] == seq2))
                return 1;
        }
    }
    return 0;
}

__global__ void determinePartialScores(char *baseSeq, char *mutation, int *cmpRes, int *weights, int numOfChecks)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= numOfChecks)
        return;
    if (tid < numOfChecks)
    {
        #ifdef DEBUG
        // Check all passed parameters and their assignment
        printf("base: ");
        for(int i =0; i< numOfChecks; i++){
            printf("%c", baseSeq[i]);
        }
        printf(" . mutation: ");
        for(int i =0; i< numOfChecks; i++){
            printf("%c", mutation[i]);
        }
        printf(" weights:");
        for(int i = 0; i<4; i++){
            printf("%d", weights[i]);
        }
	    // printf("tid: %d, base:%c, mut: %c\n", tid, baseSeq[tid], mutation[tid]);
        #endif

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

        #ifdef DEBUG
        // Check result in device
        printf(" cmp: ");
        for (int i = 0; i < numOfChecks; i++)
        {
            printf("%d,", cmpRes[i]);
        }
        printf("\n");
        #endif
    }
}

void checkError(cudaError_t cudaError, const char* s_err)
{
    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "%s - %s\n", s_err, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }
}

void launchCuda(char *baseSeq, char *mutation, int lenOfAugmented, int *cmpRes, int *weights)
{
    // Error code to check return values for CUDA calls
    cudaError_t cudaError = cudaSuccess;
    char *cuda_baseSeq;
    char *cuda_mutation;
    int *cuda_cmpRes;
    int *cuda_weights;
    
    // Allocate memory on GPU
    cudaError = cudaMalloc((void **)&cuda_baseSeq, lenOfAugmented);
    checkError(cudaError, "Failed to allocate device memory seq2");

    cudaError = cudaMalloc((void **)&cuda_mutation, lenOfAugmented);
    checkError(cudaError, "Failed to allocate device memory seq1");

    cudaError = cudaMalloc((void **)&cuda_cmpRes, lenOfAugmented * sizeof(int));
    checkError(cudaError, "Failed to allocate device memory w_cuda-");

    cudaError = cudaMalloc((void **)&cuda_weights, WEIGHTS * sizeof(int));
    checkError(cudaError, "Failed to allocate device memory w_cuda-");

    // Copy from host to device
    cudaError = cudaMemcpy(cuda_baseSeq, baseSeq, lenOfAugmented, cudaMemcpyHostToDevice);
    checkError(cudaError, "Failed to copy data from host to device seq2 -");

    cudaError = cudaMemcpy(cuda_mutation, mutation, lenOfAugmented, cudaMemcpyHostToDevice);
    checkError(cudaError, "Failed to copy data from host to device seq1 -");

    cudaError = cudaMemcpy(cuda_weights, weights, WEIGHTS * sizeof(int), cudaMemcpyHostToDevice);
    checkError(cudaError, "Failed to copy data from host to device w_cuda -");

    // Calculate the number of blocks
    int blocksPerGrid = (lenOfAugmented + MAX_THREADS - 1) / MAX_THREADS;
    
    #ifdef DEBUG_FINAL
    // printf("The number of blocks per grid is: %d\n", blocksPerGrid);
    printf("mutation: ");
    for (int i = 0; i < lenOfAugmented; i++)
    {
        printf("%c", mutation[i]);
    }
    printf("\n");
    #endif

    // Launch the Kernel
    determinePartialScores<<<blocksPerGrid, MAX_THREADS>>>(cuda_baseSeq, cuda_mutation, cuda_cmpRes, cuda_weights, lenOfAugmented);
    cudaError = cudaDeviceSynchronize();
    checkError(cudaError, "Failed to synch kernel -");

    cudaError = cudaGetLastError();
    checkError(cudaError, "Failed kernel -");
    
    // Copy results
    cudaError = cudaMemcpy(cmpRes, cuda_cmpRes, lenOfAugmented * sizeof(int), cudaMemcpyDeviceToHost);
    checkError(cudaError, "Failed to copy data device to host results -");
    
    #ifdef DEBUG
    printf(" cmp: ");
    for (int i = 0; i < lenOfAugmented; i++)
    {
        printf("%d,", cmpRes[i]);
    }
    printf("\n");
    printf("tid: %d, base:%c, mut: %c\n", tid, baseSeq[tid], mutation[tid]);
    #endif

    cudaError = cudaFree(cuda_baseSeq);
    checkError(cudaError, "Failed to free base sequence -");

    cudaError = cudaFree(cuda_mutation);
    checkError(cudaError, "Failed to free mutation sequence -");

    cudaError = cudaFree(cuda_cmpRes);
    checkError(cudaError, "Failed to free results -");   

    cudaError = cudaFree(cuda_weights);
    checkError(cudaError, "Failed to free weights -");
    
    return;
}


