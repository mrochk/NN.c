#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "../vector/vector.h"
#include "../matrix/matrix.h"
#include "../../activations/activations.h"

/* returns the dot product between v and w */
float dotprod(Vector v, Vector w) {
    assert(v->n == w->n);

    float dot = 0.0f;
    for (int i = 0; i < v->n; i++) { dot += v->d[i] * w->d[i]; }

    return dot;
}

/* compute y = A@v where A is (m * n) and v is (n * 1) */
void matvecmul(Matrix A, Vector v, Vector r) {
    assert(v->n == A->n); assert(r->n == A->m);

    for (int i = 0; i < A->m; i++) { r->d[i] = dotprod(v, A->d[i]); }

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
    assert(v); assert(w);
    assert(v->n == w->n);

    for (int i = 0; i < v->n; i++) { v->d[i] = v->d[i] + w->d[i]; }

    return;
}

/* v_i = f(v_i) for all v_i in v (inplace op) */
void vector_apply(Vector v, Activation f) {
    if (f == Identity) { return; }

    for (int i = 0; i < v->n; i++) {
        float x = v->d[i];

        switch (f) {
            case Sigmoid: 
                v->d[i] = sigmoid(x); 
                break;

            case ReLU: 
                v->d[i] = relu(x);
                break;

            case Tanh: 
                v->d[i] = tanhf(x);
                break;

            default: 
                assert(0 && "error");
        }
    }

    return;
}

/* m_{i,j} = f(m_{i,j}) for all m_{i,j} in M (inplace op) */
void matrix_apply(Matrix M, Activation f) {
    assert(M);

    for (int i = 0; i < M->m; i++) { vector_apply(M->d[i], f); }

    return;
}

/* forall M_i in M, M_i += v with |v| = |M[i]| */
void matrix_add_vector(Matrix M, Vector v) {
    assert(M); assert(v); assert(M->n == v->n);

    for (int i = 0; i < M->m; i++) { vector_add(M->d[i], v); }

    return;
}

/* if M is (1 x n) or (m x 1), copy M into v */
void vector_copy_matrix(Vector v, Matrix M) {
    assert(M); assert(v);
    assert(M->m == 1 || M->n == 1);

    if (M->m == 1) {
        assert(M->n == v->n);

        vector_copy(v, M->d[0]);

        return;
    }

    // else
    assert(M->m == v->n);

    for (int i = 0; i < M->m; i++) { v->d[i] = M->d[i]->d[0]; }

    return;
}
