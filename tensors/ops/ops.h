#ifndef OPS_H
#define OPS_H

#include "../vector/vector.h"
#include "../matrix/matrix.h"
#include "../../activations/activations.h"

float dotprod(Vector v, Vector w);

void matmul(Matrix A, Matrix B, Matrix C);

void matvecmul(Matrix M, Vector v, Vector y);

void vector_add(Vector v, Vector w);

void vector_apply(Vector v, ActivationFunc f);

void matrix_apply(Matrix M, ActivationFunc f);

/* forall i, M[i] = M[i] + v, where |v| = |M[i]| */
matrix_add_vector(Matrix M, Vector v);

#endif