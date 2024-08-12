#ifndef MLP_H
#define MLP_H

#include "../../tensors/tensors.h"
#include "../../activations/activations.h"
#include "../../utils/utils.h"

struct Layer_t {
    int inputs;
    int outputs;
    Matrix weights;
    Vector biases;
    ActivationFunc activation;
};

typedef struct Layer_t* Layer;

Layer layer_new_(int inputs, int outputs, ActivationFunc activation);

void layer_free(Layer layer);

Vector layer_forward(Layer layer, Vector x, Vector r);

struct NN_t {
    int inputs;
    int outputs;
    int hidden_layers;
};

typedef struct NN_t* NN;

#endif