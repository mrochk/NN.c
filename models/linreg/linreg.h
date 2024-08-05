#ifndef LINREG_H
#define LINREG_H

#include <stdlib.h>
#include <stdio.h>

#include "../../tensors/tensors.h"

typedef struct {
    Vector weights;
    float bias;
    float lr;
} Linreg_t;

typedef Linreg_t* Linreg;

Linreg linreg_new_(float lr, int n_features);

void linreg_free(Linreg linreg);

float linreg_predict(Linreg linreg, Vector x);

Vector linreg_predict_batch(Linreg linreg, Matrix x, Vector preds);

#endif