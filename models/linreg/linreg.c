#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "../../utils/utils.h"
#include "../../tensors/tensors.h"

#include "linreg.h"

Linreg linreg_new_(float lr, int n_features) {
    Linreg linreg = (Linreg) malloc(sizeof(Linreg));

    /* learning rate init */
    linreg->lr = lr;
    /* weights init */
    linreg->weights = vector_new_randfloat_(n_features);
    /* same for bias */
    linreg->bias = 0.0f;

    return linreg;
}

void linreg_free(Linreg linreg) {
    vector_free(linreg->weights);
    free(linreg);
}

float linreg_predict(Linreg linreg, Vector x) {
    assert(x && linreg);
    assert(x->n > 0);
    assert(linreg->weights->n == x->n);

    float pred = dotprod(linreg->weights, x) + linreg->bias;

    return pred;
}

Vector linreg_predict_batch(Linreg linreg, Matrix X, Vector preds) { assert(preds && X && linreg);
    assert(X->m > 0);
    assert(linreg->weights->n == X->d[0]->n);

    for (int i = 0; i < X->m; i++) {
        preds->d[i] = dotprod(linreg->weights, X->d[i]) + linreg->bias;
    }

    return preds;
}