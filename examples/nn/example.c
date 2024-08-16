#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "../../loss/loss.h"
#include "../../utils/utils.h"
#include "../../tensors/tensors.h"

#include "../../models/nn/nn.h"

void neuralnet_run_eg(int iters) {
    /*** create synthetic data using f(x, y, z) = sin(x) + sin(2y) + z ***/
    Matrix X = matrix_new_randfloat_(100, 3);

    Vector y = vector_new_(100);
    for (int i = 0; i < y->n; i++) {
        Vector row = X->d[i];
        float x1 = row->d[0], x2 = row->d[1], x3 = row->d[2];
        y->d[i] = sinf(x1) + sinf(2.F*x2) + x3;
    }

    puts("X:");
    matrix_print(X);

    puts("y:");
    vector_print(y);

    /*** init model ***/

    Pair arch[] = {pair(3, 10), pair(10, 10), pair(10, 1)};

    NN nn = nn_new_(3, arch, ReLU);

    /*** initial preds ***/

    Vector preds = vector_new_(100);
    Matrix P = matrix_new_(100, 1);

    nn_forward_batch(nn, X, P); vector_copy_matrix(preds, P);

    puts("\ninitial predictions:");
    vector_print(preds);

    float init_loss = MSE(preds, y);
    printf("\ninitial loss: %.8f\n\n", init_loss);

    /*** training loop ***/

    float loss = 0.F;
    float eps = 0.00001F;
    float lr = 0.001F;

    for (int iter = 0; iter < iters; iter++) {
        /* forward */
        nn_forward_batch(nn, X, P); vector_copy_matrix(preds, P);

        loss = MSE(preds, y);
        printf("-/ iter %d, loss: %.2f\n", iter, loss);

        /* backward */

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
                nn_forward_batch(nn, X, P); vector_copy_matrix(preds, P);
                float temploss = MSE(preds, y);
                db->d[i] = (temploss - loss) / eps;
                b->d[i] -= eps;
            }

            for (int i = 0; i < W->m; i++) {
                for (int j = 0; j < W->n; j++) {
                    dw->d[i]->d[j] += eps;
                    nn_forward_batch(nn, X, P); vector_copy_matrix(preds, P);
                    float temploss = MSE(preds, y);
                    dw->d[i]->d[j] = (temploss - loss) / eps;
                    dw->d[i]->d[j] -= eps;
                }
            }

            dws[l] = dw;
            dbs[l] = db;
        }

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
            matrix_free(dws[l]);
            vector_free(dbs[l]);
        }
    }

    nn_forward_batch(nn, X, P); vector_copy_matrix(preds, P);
    loss = MSE(preds, y);
    printf("\ninitial loss: %.8f\n", init_loss);
    printf("final loss:   %.8f\n", loss);

    matrix_free(X); matrix_free(P);
    vector_free(y); vector_free(preds);
}