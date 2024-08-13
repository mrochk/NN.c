#ifndef OPS_H
#define OPS_H

#include "../vector/vector.h"
#include "../matrix/matrix.h"
#include "../../activations/activations.h"

float dotprod(Vector v, Vector w);

void matrix_vector_mul(Matrix M, Vector v, Vector y);

void vector_add(Vector v, Vector w);

void vector_apply(Vector v, ActivationFunc f);

#endif