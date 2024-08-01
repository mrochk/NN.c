#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include "../utils/utils.h"
#include "../tensors/tensors.h"
#include "../activations/activations.h"

#include "neuron.h"

Neuron new_neuron_(int n_inputs, Activation activation) {
    Neuron neuron = (Neuron) malloc(sizeof(Neuron));
    neuron->w = new_random_float_vector_(n_inputs);
    neuron->b = randfloat();
    neuron->activation = activation;

    
    return neuron;
}

void free_neuron(Neuron n) {
    free_vector(n->w);
    free(n);
}

float neuron_forward(Neuron n, Vector x) {
    /* z = x.T @ w + b */
    float z = dotprod(x, n->w) + n->b;

    switch (n->activation) {
        case Linear:  return z;
        case Sigmoid: return sigmoid(z);
        case ReLU:    return relu(z);
        default:      return -1.0f;
    }
}