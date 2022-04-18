#ifndef CUDA_FUNCS_H
#define CUDA_FUNCS_H
#include "data.h"

void launchCuda(char *baseSeq, char *mutation, int lenOfAugmented, int *cmpRes, int* weights);

#endif