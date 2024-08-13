#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "../vector/vector.h"
#include "../matrix/matrix.h"
#include "../../activations/activations.h"

/* calculate the dot product between two vectors */
float dotprod(Vector v, Vector w) {
    assert(v->n == w->n);

    int N = v->n;
    float dot = 0.0f;
    for (int i = 0; i < N; i++) {
        dot += v->d[i] * w->d[i];
    }

    return dot;
}

/* compute r = A@v where A is (m * n) and v is (n * 1) */
void matrix_vector_mul(Matrix A, Vector v, Vector r) {
    assert(v->n == A->n); assert(r->n == A->m);

    for (int i = 0; i < A->m; i++) {
        r->d[i] = dotprod(v, A->d[i]);
    }
}

/* compute v + w where dim(v) = dim(w), store the result in v */
void vector_add(Vector v, Vector w) {
    assert(v->n == w->n);

    for (int i = 0; i < v->n; i++) {
        v->d[i] = v->d[i] + w->d[i];
    }
}

void vector_apply(Vector v, ActivationFunc f) {
    if (f == Linear) { return; }

    for (int i = 0; i < v->n; i++) {
        float x = v->d[i];
        switch (f) {
            case Sigmoid: v->d[i] = sigmoid(x); break;
            case ReLU:    v->d[i] = relu(x);    break;
            case Tanh:    assert(0 && "not impl");
            default:      assert(0 && "error");
        }
    }
}

