#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "mpi.h"
#include "funcs.h"
#include "data.h"


// #define DEBUG 1
// #define DEBUG2 1
// #define DEBUG3 1

enum task
{
    WORK,
    RESULT_SCORE,
    STOP
};




// typedef struct
// {
//     char *seq1;
//     int seq1_len;
//     int number_of_sequences;
//     char **sequences; // Seq2's
//     int *sequences_len;
// } Input;

int main(int argc, char *argv[])
{
    int rank, size;
    int scoringWeights[WEIGHTS];
    char buffer[MAX_LEN];
    
    char baseSeq[MAX_LEN];
    int baseSeqLen;

    char maxCmpSeq[MAX_CMP_SEQ];
    
    char **cmpSeqs;
    int *cmpSeqsLengths;
    int numOfcmpSeqs;
    
    // Variables for output
    int *offsets;
    int *n;
    int *k;

    MPI_Status status;

    MPI_Datatype mpi_results;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == ROOT)
    {
        // Result tempResult;
        // tempResult.score = 1000000;
        // // tempResult.score = 10;
        // // int numOfMutations = 0;
        // // #pragma omp parallel for private(numOfMutations) task_reduction(+: numOfMutations)
        // #pragma omp paraller for private(tempResult) task_reduction(max: tempResult.score)
        //     for (int i = 0; i < 1000; i++){
        //     // tempResult.score = i;
        //     tempResult.n = i;
        //     // numOfMutations++;
        // }
        // printf("%d\n", tempResult.score);
        // printf("%d\n", tempResult.n);

        // char *bla = "abcdefgh";
        // char *temp = (char *)calloc(3, sizeof(char));
        // int n2 = 7;
        // int k2 = 8;
        // memcpy(temp, bla, n2 - 1);
        // memcpy(temp + n2 - 1, bla + n2, k2 - n2 - 1);
        // memcpy(temp + k2 - 2, bla + k2, strlen(bla) - k2);
        // for(int i = 0; i < strlen(temp); i++){
        //     printf("%c", temp[i]);
        // }
        // printf("\n");

        // Start reading input file from input stream (input is piped into the program)
        fgets(buffer, MAX_LEN, stdin);

        // Read the weights according to given input format
        sscanf(buffer, "%d %d %d %d", &scoringWeights[0], &scoringWeights[1], &scoringWeights[2], &scoringWeights[3]);

        // Send scoring weights to all worker process for score calculations
        MPI_Bcast(&scoringWeights, WEIGHTS, MPI_INT, ROOT, MPI_COMM_WORLD);

        // Read and load base sequence (Seq1)
        fgets(buffer, MAX_LEN, stdin);
        sscanf(buffer, "%s", baseSeq);

        // Determine length of base sequence. 
        baseSeqLen = strlen(baseSeq);

        #ifdef DEBUG
        for (int i = 0; i < baseSeqLen; i++)
        {
            printf("%c", baseSeq[i]);
        }
        printf("\n");
        #endif

        // Read and load number of sequences to be compared
        fgets(buffer, MAX_LEN, stdin);
        sscanf(buffer, "%d", &numOfcmpSeqs);

        // Init results arrays
        offsets = (int *)calloc(numOfcmpSeqs, sizeof(int));
        n = (int *)calloc(numOfcmpSeqs, sizeof(int));
        k = (int *)calloc(numOfcmpSeqs, sizeof(int));

        // Allocate memory for comparison sequences
        cmpSeqs = (char **)calloc(numOfcmpSeqs, sizeof(char*));

        // Allocate memory comparison sequences' lengths
        cmpSeqsLengths = (int *)calloc(numOfcmpSeqs, sizeof(int));

        // Read and load sequences
        for (int i = 0; i < numOfcmpSeqs; i++)
        {
            // Load comparison sequence
            fgets(buffer, MAX_IN, stdin);
            sscanf(buffer, "%s", maxCmpSeq);

            // Determine and save length of sequence
            int cmpLen = strlen(maxCmpSeq);
            cmpSeqsLengths[i]=cmpLen;

            // Save comparison sequence
            cmpSeqs[i] = (char *)calloc(cmpLen, sizeof(char));
            if (!cmpSeqs[i]){
               printf("Failed to allocate memory for comparison sequence");
               return 1;
            }

            strcpy(cmpSeqs[i], maxCmpSeq);

            #ifdef DEBUG
            printf("len of input %d\n", cmpSeqsLengths[i]);
            for(int j = 0; j < cmpLen; j++){
                printf("%c", cmpSeqs[i][j]);
            }
            printf("\n");
            #endif
        }

        #ifdef DEBUG
        printf("%p\n",cmpSeqsLengths);
        for (int k=0; k < numOfcmpSeqs; k++){
            printf("len of seq %d\n", cmpSeqsLengths[k]);
             for (int j = 0; j < cmpSeqsLengths[k]; j++)
             {
                 printf("%c", cmpSeqs[k][j]);
             }
             printf("\n");
        }
        printf("base sequence length %d\n", baseSeqLen);
        #endif

        // double t = MPI_Wtime();
        
        // Dynamic work allocation - similar structure to the first HW1 assignment.
        int jobs_sent = 0;
        for (int worker_id = 1; worker_id < size; worker_id++)
        {
            // Send base sequence length
            MPI_Send(baseSeq, baseSeqLen, MPI_CHAR, worker_id, WORK, MPI_COMM_WORLD);
            // Send base sequence itself
            MPI_Send(cmpSeqs[worker_id - 1], cmpSeqsLengths[worker_id - 1], MPI_CHAR, worker_id, WORK, MPI_COMM_WORLD);
            // Increase sent jobs
            jobs_sent++;
        }

        // This is a temporary storage for our results. It will hold the offset, n, and k integers from the worker process
        int *res = (int *)malloc(sizeof(int) *3);

        #ifdef DEBUG
        printf("jobs sent: %d\n", jobs_sent);
        #endif

        for (int jobs_done = 0; jobs_done <= numOfcmpSeqs-1; jobs_done++)
        {
            #ifdef DEBUG
            printf("jobs done: %d\n", jobs_done);
            #endif
            MPI_Recv(res, 3, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            #ifdef DEBUG
            printf("received res in root from %d\n", status.MPI_SOURCE);
            #endif

            #ifdef DEBUG3
            printf("Saving results from: %d\n", status.MPI_SOURCE);
            printf("%p \n", res);
            printf("%p\n", offsets);
            printf("%p\n", n);
            printf("%p\n", k);
            // for (int i = 0; i < 3; i++)
            // {
            //     offsets[i] = res[i];
            // }
            // for (int i = 0; i < 3; i++)
            // {
            //     printf("%d\n", offsets[i]);
            // }
            #endif

            // Save results here
            offsets[status.MPI_SOURCE] = res[0];
            n[status.MPI_SOURCE] = res[1];
            k[status.MPI_SOURCE] = res[2];

            #ifdef DEBUG
            printf("Saved results\n");
            printf("%d\n", offsets[status.MPI_SOURCE]);
            printf("%d\n", n[status.MPI_SOURCE]);
            printf("%d\n", k[status.MPI_SOURCE]);
            #endif

            int jobs_left = numOfcmpSeqs - jobs_sent;
            if (jobs_left > 0)
            {   

                #ifdef DEBUG
                printf("Sending new job to %d. Sending the %dth secquence", status.MPI_SOURCE, jobs_sent + 1);
                #endif

                MPI_Send(cmpSeqs[jobs_sent], cmpSeqsLengths[jobs_sent], MPI_CHAR, status.MPI_SOURCE, WORK, MPI_COMM_WORLD);
                jobs_sent++;

                #ifdef DEBUG
                printf("send one more job, total sent is now %d\n", jobs_sent);
                #endif

            }
            else
            {
                /* send STOP message. message has no data */
                char dummy;
                MPI_Send(&dummy, 0, MPI_CHAR, status.MPI_SOURCE, STOP, MPI_COMM_WORLD);
                #ifdef DEBUG
                printf("sent STOP to process %d\n", status.MPI_SOURCE);
                #endif
            }
        }
        #ifdef DEBUG
        printf("Exited recv loop in root\n");
        #endif
        // Write outputfile

        // Free all variables.
    }
    else
    {
        // Receive scoring weights
        MPI_Bcast(&scoringWeights, 4, MPI_INT, ROOT, MPI_COMM_WORLD);

        // Probe example from Rookie HPC 
        // Prepare to receive base sequence
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // Get the number of chars in the message probed
        int baseSeqLen;
        MPI_Get_count(&status, MPI_CHAR, &baseSeqLen);

        // Allocate the buffer now that we know how many elements there are
        char *baseSeq = (char *)calloc(baseSeqLen, sizeof(char));

        // Finally receive the message
        MPI_Recv(baseSeq, baseSeqLen, MPI_CHAR, 0, WORK, MPI_COMM_WORLD, &status);

        #ifdef DEBUG
        printf("Printing base seq in %d\n", rank);
        for (int i = 0; i < baseSeqLen; i++)
        {
            printf("%c", baseSeq[i]);
        }
        printf("\n");
        #endif

        // Receive sequences for processing until a stop tag is received
        int tag;
        while(1) {
            // Prepare to receive comparative sequence
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            // Get the number of chars in the message probed
            int cmpSeqLen;
            MPI_Get_count(&status, MPI_CHAR, &cmpSeqLen);

            // Allocate the buffer now that we know how many elements there are
            char *cmpSeq = (char *)malloc(sizeof(char) * cmpSeqLen);

            // Finally receive the message
            MPI_Recv(cmpSeq, cmpSeqLen, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == STOP){
                break;
            }

            #ifdef DEBUG
            printf("Printing cmp seq in %d\n", rank);
            for (int i = 0; i < baseSeqLen; i++)
            {
                printf("%c", cmpSeq[i]);
            }
            printf("\n");
            #endif


            #ifdef DEBUG
            printf("Sending message from rank %d\n", rank);
            #endif
            // Result *result = (Result *)calloc(1, sizeof(Result));
            int *result = (int *)calloc(3, sizeof(int));
            // int result[3] = {1,2,3};
            #ifdef DEBUG
            printf("%p\n", &result);
            for(int i = 0; i < 3; i++){
                result[i] = i+10;
            }
            // for(int i = 0; i < 3; i++){
            //     printf("%d\n", result[i]);
            // }
            #endif
            
            findOptimalMutationOffset(baseSeq, cmpSeq, baseSeqLen, cmpSeqLen, scoringWeights, result);
            MPI_Send(result, 3, MPI_INT, ROOT, RESULT_SCORE, MPI_COMM_WORLD);

            #ifdef DEBUG
            printf("Sent message from rank %d\n", rank);
            #endif
        } 

        #ifdef DEBUG
        printf("Exited work in rank %d\n", status.MPI_SOURCE);
        #endif
        
        // Send results to root
        // MPI_Send(results, input.number_of_sequences, mpi_results, ROOT, RESULT_SCORE, MPI_COMM_WORLD);

        // free(results);
        // free(input.seq1);
        // free(input.sequences);
        // free(input.sequences_len);
    }

    // MPI_Type_free(&mpi_results);
    MPI_Finalize();
}