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
void matvecmul(Matrix A, Vector v, Vector r) {
    assert(v->n == A->n); assert(r->n == A->m);

    for (int i = 0; i < A->m; i++) {
        r->d[i] = dotprod(v, A->d[i]);
    }

    return;
}

/* compute C = A@B.T where A is (m * n) and B.T is (n * k) */
void matmul(Matrix A, Matrix B, Matrix C) {
    assert(A->n == B->n); 
    assert(C->m == A->m);
    assert(C->n == B->m);

    for (int i = 0; i < C->m; i++) {
        for (int j = 0; j < C->n; j++) {
            C->d[i]->d[j] = dotprod(A->d[i], B->d[j]);
        }
    }
    
    return;
}

/* compute v + w where dim(v) = dim(w), store the result in v */
void vector_add(Vector v, Vector w) {
    assert(v->n == w->n);

    for (int i = 0; i < v->n; i++) {
        v->d[i] = v->d[i] + w->d[i];
    }

    return;
}

void vector_apply(Vector v, ActivationFunc f) {
    if (f == Identity) { return; }

    for (int i = 0; i < v->n; i++) {
        float x = v->d[i];
        switch (f) {
            case Sigmoid: v->d[i] = sigmoid(x); break;
            case ReLU:    v->d[i] = relu(x);    break;
            case TanH:    assert(0 && "not impl");
            default:      assert(0 && "error");
        }
    }

    return;
}

void matrix_apply(Matrix M, ActivationFunc f) {
    for (int i = 0; i < M->m; i++) {
        vector_apply(M->d[i], f);
    }

    return;
}

/* forall i, M[i] = M[i] + v, where |v| = |M[i]| */
matrix_add_vector(Matrix M, Vector v) {
    assert(M); assert(v);
    assert(M->n == v->n);

    for (int i = 0; i < M->m; i++) { vector_add(M->d[i], v); }

    return;
}

