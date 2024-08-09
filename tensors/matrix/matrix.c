#include <stdlib.h>
#include <stdio.h>

#include "matrix.h"
#include "../vector/vector.h"
#include "../../utils/utils.h"

Matrix matrix_new_(int m, int n) {
    Matrix matrix = (Matrix) malloc(sizeof(matrix));
    matrix->m = m;
    matrix->n = n;
    matrix->d = (Vector*) malloc(sizeof(Vector*) * m);
    for (int i = 0; i < m; i++) {
        matrix->d[i] = vector_new_(n);
    }

    return matrix;
}

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

Matrix matrix_from_vectors(int n, Vector* vectors, int axis) {
    assert(n > 0 && "can't create a matrix from 0 vectors");
    assert((axis == 0 || axis == 1) && "can only concatenate on axis 0 or 1");

    size_t len = vectors[0]->n;
    for (int i = 0; i < n; i++) {
        assert(vectors[i]->n == len && "all vectors must be of same size");
    }

    if (axis == 0) {
        Matrix M = matrix_new_zeros_(n, len);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < len; j++) {
                M->d[i]->d[j] = vectors[i]->d[j];
            }
        }
        return M;
    }
    
    // elif axis == 1
    Matrix M = matrix_new_zeros_(len, n);
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < n; j++) {
            M->d[i]->d[j] = vectors[i]->d[j];
        }
    }
    return M;
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