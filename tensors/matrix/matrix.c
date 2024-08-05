#include <stdlib.h>
#include <stdio.h>

#include "matrix.h"
#include "../vector/vector.h"
#include "../../utils/utils.h"

Matrix matrix_new_zeros_(int m, int n) {
    Matrix matrix = (Matrix) malloc(sizeof(matrix));

    matrix->m = m;
    matrix->n = n;
    
    matrix->d = (Vector*) malloc(sizeof(Vector*) * m);

    for (int i = 0; i < m; i++) {
        matrix->d[i] = vector_new_zeros_(n);
    }

    return matrix;
}

Matrix matrix_new_randfloat_(int m, int n) {
    Matrix matrix = (Matrix) malloc(sizeof(matrix));

    matrix->m = m;
    matrix->n = n;
    
    matrix->d = (Vector*) malloc(sizeof(Vector*) * m);

    for (int i = 0; i < m; i++) {
        matrix->d[i] = vector_new_randfloat_(n);
    }

    return matrix;
}

Matrix matrix_new_randint_(int m, int n, int high) {
    Matrix matrix = (Matrix) malloc(sizeof(matrix));

    matrix->m = m;
    matrix->n = n;
    
    matrix->d = (Vector*) malloc(sizeof(Vector*) * m);

    for (int i = 0; i < m; i++) {
        matrix->d[i] = vector_new_randint_(n, high);
    }

    return matrix;
}

void matrix_free(Matrix M) {
    for (int i = 0; i < M->m; i++) { 
        vector_free(M->d[i]);
    }
    free(M->d); M->d = NULL;
    free(M); M = NULL;
}

void matrix_print(Matrix matrix) {
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