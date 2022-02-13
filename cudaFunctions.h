#ifndef CUDA_FUNCS_H
#define CUDA_FUNCS_H

int launch_cuda(char* seq1, int seq1_len, char* seq2, int seq2_len ,int num_of_checks, int w[4], Result* results);

#endif