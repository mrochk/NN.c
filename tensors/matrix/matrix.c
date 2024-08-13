#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "matrix.h"
#include "../vector/vector.h"
#include "../../utils/utils.h"

Matrix __matrix_new_(int m, int n) {
    Matrix matrix = (Matrix) malloc(sizeof(struct Matrix_t));
    matrix->d = (Vector*) malloc(sizeof(Vector) * m);
    matrix->m = m, matrix->n = n;

    return matrix;
}

Matrix matrix_new_(int m, int n) {
    Matrix matrix = __matrix_new_(m, n);
    for (int i = 0; i < m; i++) { matrix->d[i] = vector_new_(n); }

    return matrix;
}

Matrix matrix_new_zeros_(int m, int n) {
    Matrix matrix = __matrix_new_(m, n);
    for (int i = 0; i < m; i++) { matrix->d[i] = vector_new_zeros_(n); }

    return matrix;
}

Matrix matrix_new_randfloat_(int m, int n) {
    Matrix matrix = __matrix_new_(m, n);
    for (int i = 0; i < m; i++) { matrix->d[i] = vector_new_randfloat_(n); }

    return matrix;
}

Matrix matrix_new_randint_(int m, int n, int hi) {
    Matrix matrix = __matrix_new_(m, n);
    for (int i = 0; i < m; i++) { matrix->d[i] = vector_new_randint_(n, hi); }

    return matrix;
}

Matrix matrix_from_vectors(int n, Vector* vectors, int axis) {
    assert(n > 0); assert(axis == 0 || axis == 1);

    size_t len = vectors[0]->n;
    for (int i = 0; i < n; i++) {
        assert(vectors[i]->n == len && "vectors must be of same dim");
    }

    Matrix M = NULL;

    if (axis == 0) {
        M = matrix_new_(n, len);
        for (int i = 0; i < n; i++) {
        for (int j = 0; j < len; j++) {
            M->d[i]->d[j] = vectors[i]->d[j];
        }}
    } else {
        M = matrix_new_(len, n);
        for (int i = 0; i < len; i++) {
        for (int j = 0; j < n; j++) {
            M->d[i]->d[j] = vectors[j]->d[i];
        }}
    }
    
    return M;
}

Matrix matrix_new_from_(Matrix M) {
    Matrix A = matrix_new_(M->m, M->n);
    for (int i = 0; i < M->m; i++) {
        for (int j = 0; j < M->n; j++) {
            A->d[i]->d[j] = M->d[i]->d[j];
        }
    }
    return A;
}

void matrix_copy(Matrix A, Matrix B) {
    assert(A->m == B->m);
    assert(A->n == B->n);

    for (int i = 0; i < A->m; i++) {
        for (int j = 0; j < A->n; j++) {
            B->d[i]->d[j] = A->d[i]->d[j];
        }
    }

    return;
}

void matrix_free(Matrix M) {
    for (int i = 0; i < M->m; i++) { vector_free(M->d[i]); }
    free(M->d); free(M);

    return;
}

void matrix_print(Matrix matrix) {
    printf("[");
    for (int i = 0; i < matrix->m; i++) {
        float* row = matrix->d[i]->d;

        if (i > 0) { printf(" "); }

        for (int j = 0; j < matrix->n; j++) {
            printf(j == matrix->n - 1 ? "%04.3f" : "%04.3f ", row[j]);
        }
        puts(i == matrix->m - 1 ? "]" : "");
    }

    return;
}