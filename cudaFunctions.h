#ifndef CUDA_FUNCS_H
#define CUDA_FUNCS_H
#include "data.h"

// int launch_cuda(char* seq1, int seq1_len, char* seq2, int seq2_len ,int num_of_checks, int w[4], Result* results);
int launch_cuda(char *baseSeq, char *mutation, int lenOfAugmented, int *cmpRes, int* weights);

#endif