#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "mlp.h"

Layer layer_new_(int inputs, int outputs, ActivationFunc activation) {
    Layer layer = (Layer) malloc(sizeof(Layer));

    layer->inputs  = inputs;
    layer->outputs = outputs;

    layer->weights = matrix_new_randfloat_(outputs, inputs);
    layer->biases  = vector_new_randfloat_(outputs);

    layer->activation = activation;

    return layer;
}

void layer_free(Layer layer) {
    matrix_free(layer->weights);
    vector_free(layer->biases);
    free(layer); layer = NULL;
}

Vector layer_forward(Layer layer, Vector x, Vector r) {
    assert(layer->outputs == r->n);
    assert(layer->inputs  == x->n);

    /* z = x@W + b */
    vector_matrix_mul(x, layer->weights, r);
    vector_add(r, layer->biases, r);

    /* apply activation func */
    for (int i = 0; i < layer->outputs; i++) {
        switch (layer->activation) {
            case ReLU:    r->d[i] = relu(r->d[i]); break;
            case Sigmoid: r->d[i] = sigmoid(r->d[i]); break;
            case Tanh:    assert(0 && "not implemented");
            case Linear:  break;
            default:      assert(0 && "error");
        }
    }

    return r;
}