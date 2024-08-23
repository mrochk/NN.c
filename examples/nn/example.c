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

/* since MSE computes the loss of the predictions given as a vector
   we need to convert the output predictions to a vector */
void forward(NN nn, Matrix X, Matrix temp, Vector preds) {
    nn_forward_batch(nn, X, temp); 
    vector_copy_matrix(preds, temp);

    return;
}

void compute_grads_(NN nn, Matrix X, Matrix temp, Vector preds, Vector y, 
                    float eps, Matrix* dws, Vector* dbs) {

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

            b->d[i] -= 2.F * eps;

            forward(nn, X, temp, preds);
            float temploss_minus = MSE(preds, y);

            b->d[i] += eps;

            /* symmetric derivative formula approx */
            db->d[i] = (temploss_plus - temploss_minus) / (2.f*eps);
        }

        for (int i = 0; i < W->m; i++) {
            for (int j = 0; j < W->n; j++) {
                dw->d[i]->d[j] += eps;

                forward(nn, X, temp, preds);
                float temploss_plus = MSE(preds, y);

                dw->d[i]->d[j] -= 2.F * eps;

                forward(nn, X, temp, preds);
                float temploss_minus = MSE(preds, y);

                dw->d[i]->d[j] += eps;

                /* symmetric derivative formula approx */
                dw->d[i]->d[j] = (temploss_plus - temploss_minus) / (2.f*eps);
            }
        }

        dws[l] = dw;
        dbs[l] = db;
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

    /* the architecture of the network */
    UintPair arch[] = { 
        pair(3,  10), 
        pair(10, 10), 
        pair(10, 10), 
        pair(10, 1)
    };

    NN nn = nn_new_(nlayers, arch, ReLU);

    /*** initial preds ***/

    Matrix temp  = matrix_new_(SAMPLES, 1); /* temporary matrix to be converted */
    Vector preds = vector_new_(SAMPLES);    /* vector given to MSE */

    forward(nn, X, temp, preds);

    puts("\ninitial predictions:");
    vector_print(preds);

    float init_loss = MSE(preds, y);
    printf("\ninitial loss: %.8f\n\n", init_loss);

    /*** training loop ***/

    float loss;
    float lr = 0.05F; /* learning rate */
    float eps = 1.0E-6F; /* h */

    for (int iter = 0; iter < iters; iter++) {

        /*** forward ***/

        forward(nn, X, temp, preds);

        loss = MSE(preds, y);
        printf("|- iter %d, loss: %.2f\n", iter+1, loss);

        /*** backward ***/
        Matrix dws[nn->nlayers];
        Vector dbs[nn->nlayers];

        /* calculate dW and db for each layer of the net */ 
        compute_grads_(nn, X, temp, preds, y, eps, dws, dbs);

        /* update params */ 
        nn_update(nn, dws, dbs, lr);

        /* TODO: this can be removed & replaced by a one-time alloc */ 
        for (int i = 0; i < nn->nlayers; i++) { 
            matrix_free(dws[i]); vector_free(dbs[i]); 
        }
    }

    /*** log results ***/ 
    printf("\ninitial loss: %.4f\n", init_loss);
    printf(  "final loss:   %.4f\n\n", loss);

    puts("targets[:15]:");
    for (int i = 0; i < 15; i++) { printf("%.2f ", y->d[i]); }
    puts("\n");

    puts("final preds[:15]:");
    for (int i = 0; i < 15; i++) { printf("%.2f ", preds->d[i]); }
    puts("");
    
    /* free allocated resources */
    matrix_free(X); matrix_free(temp);
    vector_free(y); vector_free(preds);
    nn_free(nn);
}
