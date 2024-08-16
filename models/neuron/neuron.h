#ifndef NEURON_H
#define NEURON_H

#include "../../tensors/tensors.h"
#include "../../activations/activations.h"
#include "../../utils/utils.h"

struct Neuron_t {
    Vector w; /* weights */
    float  b; /* bias */
    Activation activation; /* activation function */
};

typedef struct Neuron_t* Neuron; 

Neuron neuron_new_(int n_inputs, Activation activation);

void neuron_free(Neuron n);

void neurons_free(int n, Neuron* neurons);

float neuron_forward(Neuron n, Vector x);

Vector neuron_forward_batch(Neuron n, Matrix X, Vector preds);

#endif