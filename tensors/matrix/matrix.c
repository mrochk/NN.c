#include <stdlib.h>
#include <stdio.h>

#include "matrix.h"
#include "../../utils/utils.h"

Matrix new_zeros_matrix(int m, int n) {
    Matrix matrix = (Matrix) malloc(sizeof(matrix));

    matrix->m = m;
    matrix->n = n;
    
    matrix->data = (float**) malloc(sizeof(float*) * m);

    for (int i = 0; i < m; i++) {
        matrix->data[i] = (float*) malloc(sizeof(float) * n);
        float* row = matrix->data[i];

        for (int i = 0; i < n; i++) { row[i] = 0.0f; }
    }

    return matrix;
}

Matrix new_random_matrix(int m, int n) {
    Matrix matrix = (Matrix) malloc(sizeof(matrix));

    matrix->m = m;
    matrix->n = n;
    
    matrix->data = (float**) malloc(sizeof(float*) * m);

    for (int i = 0; i < m; i++) {
        matrix->data[i] = (float*) malloc(sizeof(float) * n);
    }

    for (int i = 0; i < m; i++) {
        float random = 0.f;
        for (int j = 0; j < n; j++) { 
            matrix->data[i][j] = random_float(); 
        }
    }

    return matrix;
}

void free_matrix(Matrix M) {
    for (int i = 0; i < M->m; i++) { free(M->data[i]); }
    free(M->data);
    free(M);
}

void print_matrix(Matrix matrix) {
    for (int i = 0; i < matrix->m; i++) {
        float* row = matrix->data[i];

        for (int j = 0; j < matrix->n; j++) {
            if (j == 0) { printf("|%.2f ", row[j]); } 
            else if (j == matrix->n-1) { printf("%.2f|", row[j]); } 
            else { printf("%.2f ", row[j]); }
        }
        puts("");
    }
}