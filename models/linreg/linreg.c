#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "linreg.h"
#include "../../utils/utils.h"
#include "../../tensors/tensors.h"

Linreg new_linreg(float lr, int n_features) {
    Linreg linreg = (Linreg) malloc(sizeof(Linreg));
    linreg->lr = lr;
    linreg->weights = new_random_float_vector(n_features);
    linreg->bias = randfloat();

    return linreg;
}

void free_linreg(Linreg linreg) {
    free_vector(linreg->weights);
    free(linreg);
}

Vector linreg_predict(Linreg linreg, Matrix X, Vector preds) {
    assert(preds && X && linreg);
    assert(X->m > 0);
    assert(linreg->weights->n == X->data[0]->n);

    for (int i = 0; i < X->m; i++) {
        preds->data[i] = dotprod(linreg->weights, X->data[i]) + linreg->bias;
    }

    return preds;
}

Vector compute_dw(Vector preds, Vector y, Matrix X, Vector dw) {
    for (int i = 0; i < X->m; i++) {
        float error = y->data[i] - preds->data[i];

        for (int j = 0; j < X->n; j++) {
            dw->data[j] += -2 * error * X->data[i]->data[j];
        }
    }

    for (int i = 0; i < X->n; i++) { dw->data[i] /= X->m; }

    return dw;
}

float compute_db(Vector preds, Vector y) {
    float db = 0.0f;

    for (int i = 0; i < y->n; i++) {
        float error = y->data[i] - preds->data[i];
        db += -2 * error;
    }

    return db / y->n;
}