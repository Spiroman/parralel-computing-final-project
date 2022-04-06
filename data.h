#ifndef DATA_H
#define DATA_H

#define ROOT 0
#define MAX_LEN 5000
#define MAX_CMP_SEQ 3000
#define MAX_IN 2000
#define GROUP_A_ROWS 9  // i
#define GROUP_A_COLS 5  // j - max len of str in group a
#define GROUP_B_ROWS 11 // i
#define GROUP_B_COLS 7  // j - max len of str in group b
#define WEIGHTS 4

typedef struct
{
    int n;
    int k;
    int offset;
    int score;
} Result;

#endif