#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

/* activation functions to choose from */
typedef enum {
    /* no activation function */
    Identity, 
    /* relu activation function */
    ReLU, 
    /* sigmoid activation function */
    Sigmoid, 
    /* tanh activation function */
    TanH,
} Activation;

/* relu activation function */
float relu(float x);

/* sigmoid activation function */
float sigmoid(float x);

#endif