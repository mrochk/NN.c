#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "../tensors/tensors.h"

/* 
Mean squared error loss function:
MSE(y, ŷ) = (1/n) Sum((y_i - ŷ_i)^2 for i = 1 -> N).
*/
float MSE(Vector preds, Vector targets) {
    assert(preds->n == targets->n);

    int N = preds->n;

    float sum = 0.0f;

    for (int i = 0; i < N; i++) {
        float error = targets->data[i] - preds->data[i];
        sum += error * error;
    }

    return sum / N;
}