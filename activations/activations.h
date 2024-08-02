#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

typedef enum {
    Linear, ReLU, Sigmoid, Tanh,
} ActivationFunc;

float relu(float x);

float sigmoid(float x);

#endif