#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "linreg.h"
#include "../../utils/utils.h"
#include "../../tensors/tensors.h"

Linreg new_linreg(float lr, int n_features) {
    Linreg linreg = (Linreg) malloc(sizeof(Linreg));

    /* learning rate init */
    linreg->lr = lr;
    /* weights initialized to 0 (worked better than random) */
    linreg->weights = new_zeros_vector_(n_features);
    /* same for bias */
    linreg->bias = 0.0f;

    return linreg;
}

void free_linreg(Linreg linreg) {
    free_vector(linreg->weights);
    free(linreg);
}

Vector linreg_predict_batch(Linreg linreg, Matrix X, Vector preds) { assert(preds && X && linreg);
    assert(X->m > 0);
    assert(linreg->weights->n == X->d[0]->n);

    for (int i = 0; i < X->m; i++) {
        preds->d[i] = dotprod(linreg->weights, X->d[i]) + linreg->bias;
    }

    return preds;
}

float linreg_predict(Linreg linreg, Vector x) {
    assert(x && linreg);
    assert(x->n > 0);
    assert(linreg->weights->n == x->n);

    float pred = dotprod(linreg->weights, x) + linreg->bias;

    return pred;
}

Vector compute_dw(Vector preds, Vector y, Matrix X, Vector dw) {
    for (int i = 0; i < X->m; i++) {
        float error = y->d[i] - preds->d[i];

        for (int j = 0; j < X->n; j++) {
            dw->d[j] += -2 * error * X->d[i]->d[j];
        }
    }

    for (int i = 0; i < X->n; i++) { dw->d[i] /= X->m; }

    return dw;
}

float compute_db(Vector preds, Vector y) {
    float db = 0.0f;

    for (int i = 0; i < y->n; i++) {
        float error = y->d[i] - preds->d[i];
        db += -2 * error;
    }

    return db / y->n;
}