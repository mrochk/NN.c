#ifndef NEURON_H
#define NEURON_H

#include "../tensors/tensors.h"

typedef enum {
    Linear, ReLU, Sigmoid, Tanh,
} Activation;

typedef struct {
    Vector w; /* weights */
    float  b; /* bias */
    Activation activation; /* activation function */
} Neuron_t;

typedef Neuron_t* Neuron; 

Neuron new_neuron_(int n_inputs, Activation activation);

void free_neuron(Neuron n);

float neuron_forward(Neuron n, Vector x);

#endif