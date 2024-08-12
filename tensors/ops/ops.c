#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "../vector/vector.h"
#include "../matrix/matrix.h"

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

/* compute v@A where v is (1 x n) and A is (n x m), store the result in r */
Vector vector_matrix_mul(Vector v, Matrix A, Vector r) {
    assert(v->n == A->n);

    for (int i = 0; i < A->m; i++) {
        r->d[i] = dotprod(v, A->d[i]);
    }

    return r;
}

/* compute v + w where dim(v) = dim(w), store the result in r */
Vector vector_add(Vector v, Vector w, Vector r) {
    assert(v->n == w->n);

    for (int i = 0; i < v->n; i++) {
        r->d[i] = v->d[i] + w->d[i];
    }

    return r;
}
