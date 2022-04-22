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

// void create_result_type(MPI_Datatype *mpi_results_type);
void create_mutation(char *seq, int n, int k, int len, char *mutation);
void findOptimalMutationOffset(char *baseSeq, char *cmpSeq, int baseSeqLen, int cmpSeqLen, int *weights, Result *result);

#endif