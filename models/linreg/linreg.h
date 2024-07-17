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

Linreg new_linreg(float lr, int n_features);

void free_linreg(Linreg linreg);

Vector linreg_predict(Linreg linreg, Matrix x, Vector preds);

void linreg_fit(Linreg linreg, Matrix M);

Vector compute_dw(Vector preds, Vector y, Matrix X, Vector dw);

float compute_db(Vector preds, Vector y);

#endif