#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "linreg/linreg.h"

int main(int argc, char **argv) {
    srand(time(NULL));

    // w -> 2, b -> 1;

    Linreg* model = new_linreg();

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
    float y[] = {3.0f, 5.0f, 7.0f, 9.0f, 11.0f};

    int n = 5;

    printf("w = %f\n", model->w);

    for (int i = 0; i < 1000; i++) {
        // forward pass
        float* predictions = forward(model, x, n);

        puts("preds:");
        for (int i = 0; i < n; i++) {
            printf("%f ", predictions[i]);
        }
        puts("");

        float loss = mse(predictions, y, n);
        printf("loss = %f\n", loss);

        printf("dw = %f\n", grad_w(x, y, predictions, n));

        float lr = 0.001f;

        float dw = grad_w(x, y, predictions, n);
        float db = grad_w(x, y, predictions, n);

        model->w -= lr * dw;
        model->b -= lr * db;

        printf("w = %f\n\n", model->w);
        printf("b = %f\n\n", model->b);

        free(predictions);
    }

    
    return 0;
}
