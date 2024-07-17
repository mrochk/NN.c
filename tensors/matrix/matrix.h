#ifndef MATRIX_H
#define MATRIX_H

#include "../vector/vector.h"

typedef struct {
    int m, n;
    Vector* d;
} Matrix_t;

typedef Matrix_t* Matrix;

Matrix new_zeros_matrix(int m, int n);

Matrix new_random_float_matrix(int m, int n);

Matrix new_random_int_matrix(int m, int n, int high);

void free_matrix(Matrix M);

void print_matrix(Matrix M);

#endif