#include <stdio.h>
#include "list.h"

typedef struct {
    char* file;
} io_args;

bool inline is_num_end(const int c);
char* read_num(FILE* restrict file);
void inline read_int(FILE* restrict file, int* restrict var);
void inline read_float(FILE* restrict file, float* restrict var);
struct MMOEASASolution* read_csv_base(char* restrict filename);
struct MMOEASASolution* var_io(io_args in);

#define read_csv(...) var_io((io_args){__VA_ARGS__})