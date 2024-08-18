#ifndef MATRIX_H
#define MATRIX_H

#include "../vector/vector.h"

/* collection of vectors */
struct Matrix_t {
    /* matrix dimensions */
    size_t m, n;
    /* matrix data (array of Vectors) */
    Vector* d;
};

/* collection of vectors */
typedef struct Matrix_t* Matrix;

/* create a new matrix of size (m, n) not initialized with any value */
Matrix matrix_new_(int m, int n);

/* create a new matrix of size (m, n) filled with zeros */
Matrix matrix_new_zeros_(int m, int n);

/* create a new matrix of size (m, n) filled with random floats in [0, 1] */
Matrix matrix_new_randfloat_(int m, int n);

/* create a new matrix of size (m, n) filled with random ints in [0, high] */
Matrix matrix_new_randint_(int m, int n, int high);

/* create a new matrix by concatenating vectors along the given axis */
Matrix matrix_new_from_vectors_(int n, Vector* vectors, int axis);

Matrix matrix_new_from_(Matrix M);

/* copy A in B */
void matrix_copy(Matrix A, Matrix B);

/* free a matrix allocated with new_..._matrix_ */
void matrix_free(Matrix M);

/* print a matrix one line per row */
void matrix_print(Matrix M);

#endif