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

#endif