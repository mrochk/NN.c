#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "tensors/tensors.h"
#include "utils/utils.h"
#include "models/linreg/linreg.h"
#include "loss/loss.h"

float f(float x1, float x2, float x3) {
    return 2.0f * x1 + 3.0f * x2 + 4.0f * x3;
}

int main(int argc, char** argv) {
    srand(time(NULL));

    // linear regression learns f(x1, x2, x3) = 2(x1) + 3(x2) + 4(x3)

    int n_samples  = 10;
    int n_features = 3;

    Matrix X = new_random_int_matrix(n_samples, n_features, 100); // features

    Vector y = new_zeros_vector(n_samples); // targets

    for (int i = 0; i < y->n; i++) {
        float x1 = X->data[i]->data[0], x2 = X->data[i]->data[1], x3 = X->data[i]->data[2];
        y->data[i] = f(x1, x2, x3);
    }

    puts("Features Matrix X:");
    print_matrix(X);

    puts("\nTargets Vector y:");
    print_vector(y);

    Linreg linreg = new_linreg(0.0001, n_features);

    Vector preds = new_zeros_vector(n_samples);

    Vector dw = new_zeros_vector(linreg->weights->n);

    for (int i = 0; i < 100; i++) {
        // forward
        preds = linreg_predict(linreg, X, preds);
        puts("\nPreds:");
        print_vector(preds);

        float loss = MSE(preds, y);
        printf("\nLoss: %.2f\n", loss);

        // backward
        dw = compute_dw(preds, y, X, dw);

        float db = compute_db(preds, y);

        // updating weights and bias
        for (int i = 0; i < n_features; i++) {
            linreg->weights->data[i] -= linreg->lr * dw->data[i];
        }

        linreg->bias -= linreg->lr * db;
    }

    preds = linreg_predict(linreg, X, preds);
    puts("\nFinal preds:");
    print_vector(preds);

    puts("\nTargets y:");
    print_vector(y);

    free_matrix(X);
    free_vector(y);
    free_vector(preds);
    free_vector(dw);
    free_linreg(linreg);

    return 0;
}