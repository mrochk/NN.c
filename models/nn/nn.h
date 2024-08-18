#ifndef NN_H
#define NN_H

#include "../../tensors/tensors.h"
#include "../../activations/activations.h"
#include "../../utils/utils.h"

struct Layer_t {
    uint inputs, outputs;
    /* (output x inputs) matrix */
    Matrix weights;
    /* vector of dim n = outputs */
    Vector biases;
    /* activation function */
    Activation activation;
};

typedef struct Layer_t* Layer;

Layer layer_new_(uint inputs, uint outputs, Activation activation);

void layer_free(Layer layer);

void layer_forward(Layer layer, Vector x, Vector out);

void layer_forward_batch(Layer layer, Matrix X, Matrix Out);

struct NN_t {
    uint nlayers;
    Layer* layers;
    Activation activation;
};

typedef struct NN_t* NN;

NN nn_new_(uint nlayers, UintPair* arch, Activation f); 

void nn_free(NN nn); 

void nn_forward(NN nn, Vector x, Vector o);

void nn_forward_batch(NN nn, Matrix X, Matrix Out);

void nn_update(NN nn, Matrix* dws, Vector* dbs, float lr);

#endif