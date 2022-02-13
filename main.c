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

int main(int argc, char *argv[])
{
    int rank, size;
    int weights[WEIGHTS];
    char buffer[MAX_LEN];
    int seq2_sent = 0;
    char seq[MAX_LEN];
    int seq_len, seq2_per_proc;

    Result *results;
    Input input;

    MPI_Status status;

    MPI_Datatype mpi_results;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // create data type Result for MPI
    create_result_type(&mpi_results);

    if (rank == ROOT)
    {
        // READ INPUT AND WRITE OUTPUT
        /*
        INPUT FORMAT:
        1: w1 w2 w3 24
        2: Seq1
        3: number_of_sequences
        4: seq2...
        .
        .
        number_of_sequences:
        */

        // READ VALS FOR W1 W2 W3 W4
        fgets(buffer, MAX_LEN, stdin);
        sscanf(buffer, "%d %d %d %d", &weights[0], &weights[1], &weights[2], &weights[3]);
        MPI_Bcast(&weights, WEIGHTS, MPI_INT, ROOT, MPI_COMM_WORLD);

        fgets(buffer, MAX_IN, stdin);
        sscanf(buffer, "%s", seq);

        seq_len = strlen(seq);

        input.seq1_len = seq_len;
        input.seq1 = (char *)malloc(sizeof(int) * seq_len);
        if (!input.seq1)
            return 1;
        
        strcpy(input.seq1, seq);

        // READ number_of_sequences
        fgets(buffer, MAX_LEN, stdin);
        sscanf(buffer, "%d", &input.number_of_sequences);

        // ALLOCATE MEM FOR sequences
        input.sequences = (char **)malloc(sizeof(char) * (input.number_of_sequences));

        if (!input.sequences)
            return 1;
        
        // ALLOCATE MEM FOR sequences len
        input.sequences_len = (int *)calloc(input.number_of_sequences, sizeof(int));

        if (!input.sequences_len)
            return 1;

        // READ sequences
        for (int i = 0; i < input.number_of_sequences; i++)
        {
            fgets(buffer, MAX_IN, stdin);
            sscanf(buffer, "%s", seq);

            seq_len = strlen(seq);

            input.sequences[i] = (char *)malloc(sizeof(char) * seq_len);
            if (!input.sequences[i])
                return 1;
            input.sequences_len[i] = seq_len;

            strcpy(input.sequences[i], seq);

        }

        results = (Result *)calloc(input.number_of_sequences, sizeof(Result));

        seq2_per_proc = input.number_of_sequences / (size - 1);
       
        int seq2_coeff = input.number_of_sequences % (size - 1);
     
        double t = MPI_Wtime();

        for (int i = 1; i < size; i++)
        {
            // send seq1
            MPI_Send(input.seq1, input.seq1_len, MPI_CHAR, i, WORK, MPI_COMM_WORLD);
            // send number of seq2
            MPI_Send(&seq2_per_proc, 1, MPI_INT, i, WORK, MPI_COMM_WORLD);
            // send len of seq2
            MPI_Send(input.sequences_len + seq2_sent, seq2_per_proc, MPI_INT, i, WORK, MPI_COMM_WORLD);

            //send seq2
            for (int j = 0; j < seq2_per_proc; j++)
            {
                MPI_Send(input.sequences[seq2_sent], input.sequences_len[seq2_sent], MPI_CHAR, i, WORK, MPI_COMM_WORLD);
                seq2_sent++;
            }
        }

        for (int i = 1; i < size; i++)
        {
            MPI_Recv(results, seq2_per_proc, mpi_results, i, RESULT_SCORE, MPI_COMM_WORLD, &status);
        }


        printf("Parallel time %lf\n",MPI_Wtime()-t);
        for (int i = 0; i < input.number_of_sequences; i++)
        {
            printf("best of seq%d is n=%d k=%d\n", i, results[i].n, results[i].k);
        }

        free(input.seq1);
        free(input.sequences);
        free(input.sequences_len);
        free(results);
    }
    else
    {
        MPI_Bcast(&weights, 4, MPI_INT, ROOT, MPI_COMM_WORLD);
        
        // recv seq 1
        MPI_Probe(ROOT, WORK, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_CHAR, &input.seq1_len);
        input.seq1 = (char *)malloc(sizeof(char) * (input.seq1_len + 1));
        MPI_Recv(input.seq1, input.seq1_len, MPI_CHAR, ROOT, WORK, MPI_COMM_WORLD, &status);

        // recv num of seq2s
        MPI_Recv(&input.number_of_sequences, 1, MPI_INT, ROOT, WORK, MPI_COMM_WORLD, &status);

        // alloc mem for seq2's
        input.sequences = (char **)malloc(sizeof(char) * input.number_of_sequences);
        input.sequences_len = (int *)malloc(sizeof(int) * input.number_of_sequences);

        // recv seq2 lens
        MPI_Recv(input.sequences_len, input.number_of_sequences, MPI_INT, ROOT, WORK, MPI_COMM_WORLD, &status);
        
        // alloc mem for Results
        results = (Result *)calloc(input.number_of_sequences, sizeof(Result));

        // alloc and recv mem for seq2
        for (int i = 0; i < input.number_of_sequences; i++)
        {
            input.sequences[i] = (char *)calloc(input.sequences_len[i], sizeof(char));
            MPI_Recv(input.sequences[i], input.sequences_len[i], MPI_CHAR, ROOT, WORK, MPI_COMM_WORLD, &status);
        }

        // start clacs for Sequance Aligemnt
#pragma omp parallel for
        for (int i = 0; i < input.number_of_sequences; i++)
        {
            find_best_seq_alignment(input.seq1, input.sequences[i], input.seq1_len, input.sequences_len[i], weights, &results[i]);
        }
     
        // send results to root
        MPI_Send(results, input.number_of_sequences, mpi_results, ROOT, RESULT_SCORE, MPI_COMM_WORLD);

        free(results);
        free(input.seq1);
        free(input.sequences);
        free(input.sequences_len);
    }

    MPI_Type_free(&mpi_results);
    MPI_Finalize();
}