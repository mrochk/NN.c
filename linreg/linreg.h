#ifndef LINREG_H
#define LINREG_H

#include <stdlib.h>
#include <stdio.h>

#include "../tensors/tensors.h"

typedef struct {
    Vector weights;
    float bias;
    float lr;
} Linreg_t;

typedef Linreg_t* Linreg;

Linreg new_linreg(float lr, int n_features);

void free_linreg(Linreg linreg);

float linreg_predict(Linreg linreg, Vector x);

#endif