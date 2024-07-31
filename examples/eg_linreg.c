/* Example code where we fit linear regression on f(x, y, z) = 2x + 3y + 4z. */

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#include "../tensors/tensors.h"
#include "../utils/utils.h"
#include "../loss/loss.h"

#include "../models/linreg/linreg.h"

/* data generator function */
float f(float x1, float x2, float x3) {
    float c1 = 2.0f, c2 = 3.0f, c3 = 4.0f, b = 100.0f;
    return c1 * x1 + c2 * x2 + c3 * x3 + b;
}

#define FEATURES 3
#define SAMPLES 10

void run_linreg_example(int iters) {
    srand(time(NULL));

    Matrix X = new_random_int_matrix_(SAMPLES, FEATURES, 50); /* features */

    Vector y = new_zeros_vector_(SAMPLES); /* targets */

    /* filling the features matrix */
    for (int i = 0; i < SAMPLES; i++) {
        float x1 = X->d[i]->d[0], x2 = X->d[i]->d[1], x3 = X->d[i]->d[2];
        float y_ = f(x1, x2, x3);
        float noise = randint(5) * randfloat();
        y->d[i] = randint(1) == 0 ? y_ + noise : y_ - noise;
    }

    puts("Features Matrix X:");
    print_matrix(X);

    puts("\nTargets Vector y:");
    print_vector(y);

    float learning_rate = 0.0001f;

    Linreg linreg = new_linreg(learning_rate, FEATURES);

    Vector preds = new_zeros_vector_(SAMPLES);

    Vector dw = new_zeros_vector_(linreg->weights->n);
    float  db = 0.0f;

    for (int i = 0; i < iters; i++) {

        /****** forward ******/
        preds = linreg_predict_batch(linreg, X, preds);
        float loss = MSE(preds, y);
        printf("\nLoss: %.2f", loss); fflush(stdout);

        /****** backward ******/
        preds = linreg_predict_batch(linreg, X, preds);
        dw = compute_dw(preds, y, X, dw);
        float db = compute_db(preds, y);

        /****** updating weights and bias ******/
        for (int i = 0; i < FEATURES; i++) {
            linreg->weights->d[i] -= linreg->lr * dw->d[i];
        }

        linreg->bias -= linreg->lr * db;
    }

    /****** print final predictions ******/
    preds = linreg_predict_batch(linreg, X, preds);

    puts("\n\nFinal Preds:");
    print_vector(preds);

    puts("\nTargets y:");
    print_vector(y);
    
    Vector input = new_zeros_vector_(3);
    input->d[0] = 3;
    input->d[1] = 4;
    input->d[2] = 5;

    float pred = linreg_predict(linreg, input);
    int expected = 2*3 + 3*4 + 4*5 + 100;
    printf("\nPrediction for 2*3 + 3*4 + 4*5 + 100 ~ %.2f [%d].\n", pred, expected);

    puts("\nModel Weights:");
    print_vector(linreg->weights);
    printf("\nModel Bias: %.2f.\n", linreg->bias);

    /* free memory */
    free_linreg(linreg);

    free_matrix(X);

    free_vector(y); free_vector(preds); free_vector(dw); free_vector(input);

    return;
}