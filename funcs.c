#include "funcs.h"
#include "mpi.h"

char group_one[GROUP_A_ROWS][GROUP_A_COLS] = {
    "NDEQ",
    "MILV",
    "FYW",
    "NEQK",
    "QHRK",
    "HY",
    "STA",
    "NHQK",
    "MILF"};

char group_two[GROUP_B_ROWS][GROUP_B_COLS] = {
    "SAG",
    "SGND",
    "NEQHRK",
    "HFY",
    "ATV",
    "STPA",
    "NDEQHK",
    "FVLIM",
    "CSA",
    "STNK",
    "SNDEQK"};

// Create data type Results for MPI
void create_result_type(MPI_Datatype *mpi_results_type)
{
    int blockLength[3] = {1, 1, 1}; /*number of blocks for each parameter*/
    MPI_Aint displacements[3] = {offsetof(Result, n),
                                 offsetof(Result, k),
                                 offsetof(Result, score)};

    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Type_create_struct(3, blockLength, displacements, types,
                           mpi_results_type);
    MPI_Type_commit(mpi_results_type);
}

/* Muatation creation
    Input example: char* seq = HELLO, int k = 1, int len = 5
    Output: H-ELLO
*/
char *mutation(char *seq, int k, int len)
{
    char *mutant_seq = (char *)calloc((len + 2), sizeof(char)); // memory +2 for '-' and '\0'

    if (!mutant_seq)
        return NULL;

    strncpy(mutant_seq, seq, k);
    mutant_seq[k] = '-';
    strncpy(mutant_seq + k + 1, seq + k, len - k);
    mutant_seq[len + 1] = '\0';

    return mutant_seq;
}

int compare_group_a(char seq1, char seq2)
{

    for (int i = 0; i < GROUP_A_ROWS; i++)
    {
        for (int j = 0; j < GROUP_A_COLS; j++)
        {
            if (strchr(group_one[i], seq1) && strchr(group_one[i], seq2))
                return 1;
        }
    }
    return 0;
}

int compare_group_b(char seq1, char seq2)
{
    for (int i = 0; i < GROUP_A_ROWS; i++)
    {
        for (int j = 0; j < GROUP_A_COLS; j++)
        {
            if (strchr(group_two[i], seq1) && strchr(group_two[i], seq2))
                return 1;
        }
    }
    return 0;
}

// Comapre two chars
char compare(char seq1, char seq2)
{

    if (seq1 == seq2)
    {
        return '$';
    }
    else if (compare_group_a(seq1, seq2) == 1)
    {
        return '%';
    }
    else if (compare_group_b(seq1, seq2) == 1)
    {
        return '#';
    }
    else
    {
        return ' ';
    }
}

int calc_score(char *seq1, char *seq2, int seq1_len, int seq2_len, int w[WEIGHTS])
{
    int weights[WEIGHTS] = {0};
    int alignment_score = 0;
    char compare_res;

    for (int i = 0; i < seq2_len; i++)
    {
        compare_res = compare(seq1[i], seq2[i]);
        if (compare_res == '$')
            weights[0]++;
        else if (compare_res == '%')
            weights[1]++;
        else if (compare_res == '#')
            weights[2]++;
        else
            weights[3]++;
    }
    alignment_score = (w[0] * weights[0]) - (w[1] * weights[1]) - (w[2] * weights[2]) - (w[3] * weights[3]);

    return alignment_score;
}

int find_best_seq_alignment(char *seq1, char *seq2, int seq1_len, int seq2_len, int w[WEIGHTS], Result *result)
{
    // number of checks the alogrithem has to do
    int num_of_check = seq1_len - seq2_len;

    // creare arr for mutations sequances
    char **mutation_arr = (char **)calloc(seq2_len, sizeof(char *));
    if (!mutation_arr)
        return 0;
    mutation_arr[0] = seq2;

    Result temp_result = {0, 0, 0};

#pragma omp parallel for
    for (int i = 1; i < seq2_len; i++)
    {
        mutation_arr[i] = mutation(seq2, i, seq2_len);
    }

#pragma omp parallel for private(temp_result)
    for (int i = 0; i < seq2_len; i++)
    {
        temp_result.k = i;

        launch_cuda(seq1, seq1_len, mutation_arr[i], strlen(mutation_arr[i]), num_of_check, w, &temp_result);

        // Check if current score it the max
        if (temp_result.score > result->score || result->score == 0)
        {
            result->score = temp_result.score;
            result->n = temp_result.n;
            result->k = temp_result.k;
        }
    }

    for (int i = 0; i < seq2_len; i++)
    {
        free(mutation_arr[i]);
    }

    free(mutation_arr);

    return result->score;
}