#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "tensors/tensors.h"
#include "utils/utils.h"
#include "models/linreg/linreg.h"
#include "loss/loss.h"

float f(float x1, float x2, float x3) {
    return 2.0f * x1 + 3.0f * x2 + 4.0f * x3 + 100.0f;
}

int main(int argc, char** argv) {
    srand(time(NULL));

    // linear regression learns f(x1, x2, x3) = 2(x1) + 3(x2) + 4(x3)

    int n_samples  = 10;
    int n_features = 3;

    Matrix X = new_random_int_matrix_(n_samples, n_features, 50); // features

    Vector y = new_zeros_vector_(n_samples); // targets

    for (int i = 0; i < y->n; i++) {
        float x1 = X->d[i]->d[0], x2 = X->d[i]->d[1], x3 = X->d[i]->d[2];
        y->d[i] = f(x1, x2, x3);
    }

    puts("Features Matrix X:");
    print_matrix(X);

    puts("\nTargets Vector y:");
    print_vector(y);

    Linreg linreg = new_linreg(0.0001, n_features);

    Vector preds = new_zeros_vector_(n_samples);

    Vector dw = new_zeros_vector_(linreg->weights->n);

    printf("bias before = %.4f\n", linreg->bias);

    float error_threshold = 0.1;

    int i;
    for (i = 0; 1; i++) {
        // forward
        preds = linreg_predict_batch(linreg, X, preds);
        //puts("\nPreds:");
        //print_vector(preds);

        float loss = MSE(preds, y);
        printf("\nLoss: %.2f", loss);

        if (loss < error_threshold) { break; }

        // backward
        dw = compute_dw(preds, y, X, dw);

        float db = compute_db(preds, y);

        // updating weights and bias
        for (int i = 0; i < n_features; i++) {
            linreg->weights->d[i] -= linreg->lr * dw->d[i];
        }

        linreg->bias -= linreg->lr * db;
    }

    // print final preds

    preds = linreg_predict_batch(linreg, X, preds);

    puts("\n\nFinal preds:");
    print_vector(preds);

    puts("Targets y:");
    print_vector(y);
    
    // 2*3 + 3*4 + 4*5 = 6 + 12 + 20 = 38
    Vector input = new_zeros_vector_(3);
    input->d[0] = 3;
    input->d[1] = 4;
    input->d[2] = 5;

    float pred = linreg_predict(linreg, input);

    printf("Threshold reached after %d iterations.\n", i);

    printf("2*3 + 3*4 + 4*5 + 100 = %4.f (%d)\n", pred, (2*3 + 3*4 + 4*5 + 100));

    print_vector(linreg->weights);
    printf("bias = %.4f\n", linreg->bias);

    // free memory

    free_linreg(linreg);

    free_matrix(X);

    free_vector(y);
    free_vector(preds);
    free_vector(dw);

    return 0;
}