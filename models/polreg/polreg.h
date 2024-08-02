#ifndef POLREG_H
#define POLREG_H

#include "../../tensors/tensors.h"

typedef struct {
    Vector powers;
    Vector weights;
    float  bias;
    float  lr;
} Polreg_t;

typedef Polreg_t* Polreg;

Polreg polreg_new_(Vector powers, float lr); 

void polreg_free(Polreg polreg); 

float polreg_predict(Polreg polreg, Vector x);

void polreg_predict_batch(Polreg polreg, Matrix X, Vector preds);

#endif