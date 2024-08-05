#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include "../../utils/utils.h"
#include "../../tensors/tensors.h"
#include "../../activations/activations.h"

#include "neuron.h"

Neuron neuron_new_(int n_inputs, ActivationFunc activation) {
    Neuron neuron = (Neuron) malloc(sizeof(Neuron));
    neuron->w = vector_new_randfloat_(n_inputs);
    neuron->b = randfloat();
    neuron->activation = activation;

    
    return neuron;
}

void neuron_free(Neuron n) {
    vector_free(n->w);
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