#ifndef OPS_H
#define OPS_H

#include "../vector/vector.h"
#include "../matrix/matrix.h"

float dotprod(Vector v, Vector w);

Vector vector_matrix_mul(Vector v, Matrix A, Vector r);

Vector vector_add(Vector v, Vector w, Vector r);

#endif