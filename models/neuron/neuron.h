#ifndef NEURON_H
#define NEURON_H

#include "../../tensors/tensors.h"
#include "../../activations/activations.h"

typedef struct {
    Vector w; /* weights */
    float  b; /* bias */
    ActivationFunc activation; /* activation function */
} Neuron_t;

typedef Neuron_t* Neuron; 

Neuron neuron_new_(int n_inputs, ActivationFunc activation);

void neuron_free(Neuron n);

float neuron_forward(Neuron n, Vector x);

Vector neuron_forward_batch(Neuron n, Matrix X, Vector preds);

#endif