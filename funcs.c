#include "funcs.h"
#include "mpi.h"
#include <stdio.h>
#include <limits.h>
#include "data.h"

#define DEBUG 1

// This function assumes the indices n, and k start with 1 rather than 0
void create_mutation(char *seq, int n, int k, int len, char *mutation)
{    
    // Copy the first part, up to the nth variable.
    memcpy(mutation, seq, n - 1);
    // Copy Middle part
    memcpy(mutation + n - 1, seq + n, k - n - 1);
    // Copy last part
    memcpy(mutation + k - 2, seq + k, strlen(seq) - k);
}

void find_optimal_mutation_offset(char *baseSeq, char *cmpSeq, int baseSeqLen, int cmpSeqLen, int* weights, int *result)
{
    printf("mutating\n");
    // Determine the number of offsets to try. base - comperative - 2 chars for n&k
    int numOfOffsets = baseSeqLen - cmpSeqLen + 2;
    int lenOfAugmented = cmpSeqLen - 2;
    
    // Declaration for parrallel executions
    char *mutation;
    Result tempResult;

    // We're setting the score to the lowest possible int value. 
    // Each private copy of the result will hold that score when it starts executing.
    // This ensures that all threads will catch results that are negative as well as best results. 
    // And yes, I checked it.
    tempResult.score = INT_MIN;

    // Each thread will have it's own copy of the results struct. At the end the maximum result and its values will be returned
    // In each execution we will check for each pair of mutations, for each possible offset of said mutation, what yields the best score.
    // Array indices start as 1 instead of 0, and are then adjusted in the mutation creation function.
    printf("cpm seq len: %d. numofoffsets: %d. baseseqlen: %d\n", cmpSeqLen, numOfOffsets, baseSeqLen);
    #pragma omp paraller for private(tempResult, mutation) task_reduction(max: tempResult.score)
    for (int n = 1; n < cmpSeqLen; n++){
        for (int k = n + 1; k <= cmpSeqLen; k++){
            for (int offset = 0; offset <= numOfOffsets; offset++){
                // Separate the part of the base sequence with which the comparisons will be made
                char *baseSeqAugmented = (char *)malloc(sizeof(char) * lenOfAugmented);
                memcpy(baseSeqAugmented, baseSeq + offset, lenOfAugmented);

                // Create the mutation that will be checked
                char *mutation = (char *)malloc(sizeof(char) * lenOfAugmented);
                create_mutation(cmpSeq, n, k, lenOfAugmented, mutation);
                
                #ifdef DEBUG
                for(int i = 0; i < lenOfAugmented; i++){
                    printf("%c", mutation[i]);
                }
		printf(" %s, aug base: %s, offset: %d, len of aug: %d", cmpSeq, baseSeqAugmented, offset, lenOfAugmented);
                printf("\n");
                #endif

                // Create array of partial results depending on match and its weight.
                int *cmpRes = (int *)malloc(sizeof(int) * lenOfAugmented);

                // Send mutation and comparison to be compared and determine matches (will fill the occurances int array).
                // launch_cuda(baseSeq, mutation, lenOfAugmented, cmpRes, weights);
                
                // Calculate score
                int tempScore = 0;
                for(int i = 0; i < lenOfAugmented; i++){
                    tempScore += cmpRes[i];
                }

                // Compare and assign best result
                if(tempScore > tempResult.score){
                    tempResult.score = tempScore;
                    tempResult.k = k;
                    tempResult.n = n;
                    tempResult.offset = offset;
                }
                
                // Free augmented base
                free(mutation);
                free(baseSeqAugmented);
                free(cmpRes);
            }
        }
    }
    memcpy(result, &tempResult, 1);
    return;
}
