#include <stdlib.h>
#include <stdio.h>

#include "matrix.h"
#include "../vector/vector.h"
#include "../../utils/utils.h"

Matrix new_zeros_matrix_(int m, int n) {
    Matrix matrix = (Matrix) malloc(sizeof(matrix));

    matrix->m = m;
    matrix->n = n;
    
    matrix->d = (Vector*) malloc(sizeof(Vector*) * m);

    for (int i = 0; i < m; i++) {
        matrix->d[i] = new_zeros_vector_(n);
    }

    return matrix;
}

Matrix new_random_float_matrix_(int m, int n) {
    Matrix matrix = (Matrix) malloc(sizeof(matrix));

    matrix->m = m;
    matrix->n = n;
    
    matrix->d = (Vector*) malloc(sizeof(Vector*) * m);

    for (int i = 0; i < m; i++) {
        matrix->d[i] = new_random_float_vector_(n);
    }

    return matrix;
}

Matrix new_random_int_matrix_(int m, int n, int high) {
    Matrix matrix = (Matrix) malloc(sizeof(matrix));

    matrix->m = m;
    matrix->n = n;
    
    matrix->d = (Vector*) malloc(sizeof(Vector*) * m);

    for (int i = 0; i < m; i++) {
        matrix->d[i] = new_random_int_vector_(n, high);
    }

    return matrix;
}

void free_matrix(Matrix M) {
    for (int i = 0; i < M->m; i++) { 
        free_vector(M->d[i]);
    }
    free(M->d); M->d = NULL;
    free(M); M = NULL;
}

void print_matrix(Matrix matrix) {
    printf("[");
    for (int i = 0; i < matrix->m; i++) {
        float* row = matrix->d[i]->d;

        if (i > 0) { printf(" "); }

        for (int j = 0; j < matrix->n; j++) {
            printf(j == matrix->n - 1 ? "%04.1f" : "%04.1f ", row[j]);
        }
        puts(i == matrix->m - 1 ? "]" : "");
    }
}