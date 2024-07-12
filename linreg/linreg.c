#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "linreg.h"

float init_param() {
    return (float)rand() / (float)RAND_MAX;
}

Linreg* new_linreg() {
    Linreg* linreg = (Linreg*) malloc(sizeof(Linreg));
    linreg->w = init_param();
    linreg->b = init_param();

    return linreg;
}

float* forward(Linreg* linreg, float* x, int n) {
    float* predictions = (float*) malloc(sizeof(float)*n);

    for (int i = 0; i < n; i++) {
        predictions[i] = x[i] * linreg->w + linreg->b;
    }

    return predictions;
}

float squared_error(float prediction, float target) {
    float error = target - prediction;
    return error * error; 
}

float grad_w(float* x, float* y, float* preds, int n) {
    float sum = 0.0f;

    for (int i = 0; i < n; i++) {
        sum += -2 * x[i] * (y[i] - preds[i]);
    }

    return sum / n;
}

float grad_b(float* y, float* preds, int n) {
    float sum = 0.0f;

    for (int i = 0; i < n; i++) {
        sum += -2 * (y[i] - preds[i]);
    }

    return sum / n;
}

float mse(float* preds, float* targets, int n) {
    float sum = 0.0f;

    for (int i = 0; i < n; i++) {
        float error = targets[i] - preds[i];
        sum += error * error;
    }

    return sum / n;
}