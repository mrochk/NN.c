#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#include "../tensors/tensors.h"
#include "../utils/utils.h"
#include "../loss/loss.h"

#include "../models/polreg/polreg.h"

/*
Fitting f(x) = sin(2*pi*x) using a polynomial regression model of degree 3.
Example taken from Deep Learning: Foundations & Concepts by C. Bishop.
*/
void run_polreg_example() {
    int n_features = 1;

    Vector powers = new_zeros_vector_(n_features); powers->d[0] = 3.0f;

    float learning_rate = 0.1f;

    Polreg model = polreg_new_(powers, learning_rate);

    int datapoints = 10000;

    Matrix x = new_random_float_matrix_(datapoints, n_features);

    Vector y = new_zeros_vector_(datapoints);
    for (int i = 0; i < datapoints; i++) {
        y->d[i] = sinf(2.0f * x->d[i]->d[0] * ((float)PI));
    }    

    Vector preds = new_zeros_vector_(datapoints);
    polreg_predict_batch(model, x, preds);

    printf("Predictions:");
    print_vector(preds);

    printf("Targets:");
    print_vector(y);

    printf("Model weights: ");
    print_vector(model->weights);

    printf("Model powers: ");
    print_vector(model->powers);

    printf("bias: %.2f\n", model->bias);

    printf("x: ");
    print_matrix(x);

    float error, dw, db;

    float loss = MSE(preds, y);

    printf("loss: %.2f\n", loss);

    for (int i = 0; i < 100; i++) {

        dw = 0.0F; db = 0.0F;

        for (int j = 0; j < datapoints; j++) {
            error = y->d[j] - preds->d[j];
            dw += -2.0F * powf(x->d[j]->d[0], 3.0F) * error;
            db += -2.0F * error;
        }

        dw = dw / datapoints;
        db = db / datapoints;

        model->weights->d[0] -= model->lr * dw;
        model->bias -= model->lr * db;

        polreg_predict_batch(model, x, preds);

        loss = MSE(preds, y);

        printf("iter %d, loss: %.2f\n", i+1, loss);
        puts("--------------------");
    }

    free_matrix(x);
    free_vector(preds); free_vector(y);
    polreg_free(model);
}