#ifndef DATA_H
#define DATA_H

#define ROOT 0
#define MAX_LEN 3000
#define MAX_IN 2000
#define GROUP_A_ROWS 9  // i
#define GROUP_A_COLS 5  // j - max len of str in group a
#define GROUP_B_ROWS 11 // i
#define GROUP_B_COLS 7  // j - max len of str in group b
#define WEIGHTS 4

enum task
{
    WORK,
    RESULT_SCORE,
    STOP
};

typedef struct
{
    int n; // offset
    int k; // location of hypen '-'
    int score; // score of aligment
} Result;

typedef struct
{
    char *seq1;
    int seq1_len;
    int number_of_sequences;
    char **sequences; // Seq2's
    int *sequences_len;
} Input;

#endif