#ifndef FUNCS_H
#define FUNCS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <omp.h>
#include "mpi.h"
#include "data.h"
#include "cudaFunctions.h"

void create_result_type(MPI_Datatype *mpi_results_type);
char *mutation(char *seq, int k, int len);
int find_best_seq_alignment(char *seq1, char *seq2, int seq1_len, int seq2_len, int w[WEIGHTS], Result *result);

#endif