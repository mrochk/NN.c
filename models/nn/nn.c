#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "nn.h"

Layer layer_new_(uint inputs, uint outputs, ActivationFunc activation) {
    assert(inputs > 0 && outputs > 0);

    Layer layer = (Layer) malloc(sizeof(struct Layer_t));

    layer->inputs  = inputs, layer->outputs = outputs;
    layer->weights = matrix_new_randfloat_(outputs, inputs);
    layer->biases  = vector_new_randfloat_(outputs);

    layer->activation = activation;

    return layer;
}

void layer_free(Layer layer) {
    assert(layer);

    matrix_free(layer->weights);
    vector_free(layer->biases);
    free(layer);
}

/* computing f(x@W + b) with f being the chosen activation func */
void layer_forward(Layer layer, Vector x, Vector out) {
    assert(layer->inputs  == x->n);
    assert(layer->outputs == out->n);

    /* z = x@W + b */
    matvecmul(layer->weights, x, out);
    vector_add(out, layer->biases);

    /* apply activation func */
    vector_apply(out, layer->activation);
}

/* computing f(X@W + b) with f being the chosen activation func */
void layer_forward_batch(Layer layer, Matrix X, Matrix Out) {
    assert(layer->inputs  == X->n);
    assert(layer->outputs == Out->n);

    /* Z = X@W + b */
    matmul(X, layer->weights, Out);
    for (int i = 0; i < Out->m; i++) {
        vector_add(Out->d[i], layer->biases);
    }

    /* apply activation func */
    matrix_apply(Out, layer->activation);
}


NN nn_new_(uint nlayers, Pair* structure, ActivationFunc f) {
    assert(nlayers > 0);

    NN nn = (NN) malloc(sizeof(struct NN_t));

    nn->nlayers = nlayers, nn->activation = f;

    nn->layers = (Layer*)malloc(sizeof(Layer) * nlayers);
    for (int i = 0; i < nlayers; i++) {
        int inputs  = structure[i].a, outputs = structure[i].b;

        /* no activation function for last layer */
        if (i == nlayers - 1) {
            nn->layers[i] = layer_new_(inputs, outputs, Linear);
        } else {
            nn->layers[i] = layer_new_(inputs, outputs, f);
        } 
    }

    return nn;
}

void nn_free(NN nn) {
    for (int i = 0; i < nn->nlayers; i++) {
        layer_free(nn->layers[i]);
    }
    free(nn->layers);
    free(nn);

    return;
}

Vector nn_forward(NN nn, Vector x) {
    assert(nn->layers[0]->inputs == x->n);

    for (int i = 0; i < nn->nlayers; i++) {
        Layer layer = nn->layers[i];

        Vector temp = vector_new_(layer->outputs); 

        layer_forward(layer, x, temp);

        vector_print(x);

        vector_free(x);

        x = temp;
    }

    return x;
}

void nn_forward_batch(NN nn, Matrix X, Matrix O) {
    assert(nn->layers[0]->inputs == X->n);
    
    Matrix temp1 = matrix_new_from_(X), temp2;

    for (int i = 0; i < nn->nlayers; i++) {
        Layer layer = nn->layers[i];

        /* to store the result of the layer */
        temp2 = matrix_new_(temp1->m, layer->outputs); 

        layer_forward_batch(layer, temp1, temp2);

        matrix_free(temp1);

        temp1 = temp2;
    }

    matrix_copy(temp2, O);

    matrix_free(temp2);

    return;
}