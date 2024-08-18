#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "../../loss/loss.h"
#include "../../utils/utils.h"
#include "../../tensors/tensors.h"

#include "../../models/nn/nn.h"

#define SAMPLES 100
#define FEATURES 3

void forward(NN nn, Matrix X, Matrix temp, Vector preds) {
    nn_forward_batch(nn, X, temp); 
    vector_copy_matrix(preds, temp);

    return;
}

void backward(NN nn, float eps, Matrix X, Matrix temp, Vector preds, Vector y, float lr) {
    /*** backward ***/

    Matrix dws[nn->nlayers];
    Vector dbs[nn->nlayers];

    /* compute grads */

    for (int l = 0; l < nn->nlayers; l++) {
        Layer layer = nn->layers[l];

        Matrix W = layer->weights;
        Vector b = layer->biases;

        Matrix dw = matrix_new_(W->m, W->n);
        Vector db = vector_new_(b->n);

        for (int i = 0; i < b->n; i++) {
            b->d[i] += eps;

            forward(nn, X, temp, preds);

            float temploss_plus = MSE(preds, y);

            b->d[i] -= eps;
            b->d[i] -= eps;

            forward(nn, X, temp, preds);

            float temploss_minus = MSE(preds, y);

            b->d[i] += eps;

            db->d[i] = (temploss_plus - temploss_minus) / (2.f*eps);
        }

        // [f(x - h) - f(x + h)] / 2h

        for (int i = 0; i < W->m; i++) {
            for (int j = 0; j < W->n; j++) {
                dw->d[i]->d[j] += eps;

                forward(nn, X, temp, preds);

                float temploss_plus = MSE(preds, y);

                dw->d[i]->d[j] -= eps;
                dw->d[i]->d[j] -= eps;

                forward(nn, X, temp, preds);

                float temploss_minus = MSE(preds, y);

                dw->d[i]->d[j] += eps;

                dw->d[i]->d[j] = (temploss_plus - temploss_minus) / (2.f*eps);
            }
        }

        dws[l] = dw;
        dbs[l] = db;
    }

    /* update parameters */

    for (int l = 0; l < nn->nlayers; l++) {
        Layer layer = nn->layers[l];

        Matrix W = layer->weights;
        Vector b = layer->biases;

        Matrix dw = dws[l];
        Vector db = dbs[l];

        for (int i = 0; i < b->n; i++) {
            b->d[i] -= lr * db->d[i];
        }

        for (int i = 0; i < W->m; i++) {
            for (int j = 0; j < W->n; j++) {
                W->d[i]->d[j] -= lr * dw->d[i]->d[j];
            }
        }
    }

    for (int l = 0; l < nn->nlayers; l++) {
        matrix_free(dws[l]); vector_free(dbs[l]);
    }
    
    return;
}

void neuralnet_run_eg(int iters) {

    /*** create synthetic data using f(x, y, z) = sin(pi*x) + sin(2*y) - z ***/

    Matrix X = matrix_new_randfloat_(SAMPLES, FEATURES);

    Vector y = vector_new_(SAMPLES);
    for (int i = 0; i < y->n; i++) {
        Vector row = X->d[i];
        float x1 = row->d[0], x2 = row->d[1], x3 = row->d[2];
        y->d[i] = sinf(x1*PI) + sinf(2.0f*x2) - x3;
    }

    puts("X:"); matrix_print(X);

    puts("y:"); vector_print(y);

    /*** init model ***/

    const uint nlayers = 4;

    Pair arch[] = { 
        pair(3,  10), 
        pair(10, 10), 
        pair(10, 10), 
        pair(10, 1)
    };

    NN nn = nn_new_(nlayers, arch, ReLU);

    /*** initial preds ***/

    /* since MSE computes the loss of the predictions given as a vector
       we need to convert the output predictions to a vector */

    Matrix temp  = matrix_new_(SAMPLES, 1); /* temporary matrix to be converted */
    Vector preds = vector_new_(SAMPLES); /* vector given to MSE */

    forward(nn, X, temp, preds);

    puts("\ninitial predictions:");
    vector_print(preds);

    float init_loss = MSE(preds, y);
    printf("\ninitial loss: %.8f\n\n", init_loss);

    /*** training loop ***/

    float loss;
    float lr = 0.05F;   /* learning rate */
    float eps = 1.0E-6F; /* constant term in symmetric derivative approx */

    for (int iter = 0; iter < iters; iter++) {

        /*** forward ***/

        forward(nn, X, temp, preds);

        loss = MSE(preds, y);
        printf("|- iter %d, loss: %.2f\n", iter+1, loss);

        /*** backward ***/

        backward(nn, eps, X, temp, preds, y, lr);
    }

    printf("\ninitial loss: %.8f\n", init_loss);
    printf("final loss:   %.8f\n", loss);

    puts("targets:");
    vector_print(y);

    puts("final preds:");
    vector_print(preds);

    matrix_free(X); matrix_free(temp);
    vector_free(y); vector_free(preds);
    nn_free(nn);
}