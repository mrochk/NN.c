#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "../../tensors/tensors.h"
#include "../../utils/utils.h"
#include "../../loss/loss.h"

#include "../../models/polreg/polreg.h"

void polreg_run_eg(int iters) {
    /*** create synthetic data using f(x) = sin(2*pi*x) ***/
    const int n_features = 3;
    const int datapoints = 20;

	/* features */
    Matrix X = new_random_float_matrix_(datapoints, n_features); /* features matrix */
    /* copying the first column in the second */
    for (int i = 0; i < X->m; i++) {
        X->d[i]->d[1] = X->d[i]->d[0];
        X->d[i]->d[2] = X->d[i]->d[0];
    }

	/* targets */
	Vector y = new_zeros_vector_(datapoints);
	for (int i = 0; i < y->n; i++) {
		float x = X->d[i]->d[0];
		y->d[i] = sinf(2.F * PI * x);
	}

	puts("features:");
	print_matrix(X);

	puts("\ntargets:");
	print_vector(y);

    /*** initialize the model ***/
    Vector powers = new_zeros_vector_(n_features); 
    powers->d[0] = 3.0F; /* we use a polynomial of degree 3 */
    powers->d[1] = 2.0F; powers->d[2] = 1.0F;

	Polreg model = polreg_new_(powers, .01F);

    /*** perform initial predictions ***/
	Vector predictions = new_zeros_vector_(datapoints);

	predictions = polreg_predict_batch(model, X, predictions);

	puts("\ninitial predictions:");
	print_vector(predictions);

	float init_loss = MSE(predictions, y);
	printf("\ninitial loss (mse) : %.2f\n", init_loss);

    /*** training loop ***/
	float  db = 0.F, loss;
    Vector dw = new_zeros_vector_(n_features);

	for (int i = 0; i < iters; i++) {
		/*** forward ***/
		predictions = polreg_predict_batch(model, X, predictions);

		loss = MSE(predictions, y);
        if (iters < 100) {
		    printf("iter %d, loss: %.2f\n", i, loss);
        }

		/*** backward ***/
		db = 0.F;
        for (int k = 0; k < dw->n; k++) { dw->d[k] = 0.F; }

		/* calculating dw and db */
		for (int j = 0; j < datapoints; j++) {
			float ei = y->d[j] - predictions->d[j];
            float xi = X->d[j]->d[0];

            db += -2.F * ei;
            for (int k = 0; k < dw->n; k++) { 
                dw->d[k] += -2.F * powf(xi, model->powers->d[k]) * ei; 
            }
		}

        db /= datapoints;
        for (int k = 0; k < dw->n; k++) { dw->d[k] /= datapoints; }

		/* updating weights */
		model->bias -= model->lr * db;
        for (int k = 0; k < dw->n; k++) { 
            model->weights->d[k] -= model->lr * dw->d[k];
        }
	}

	puts("\nfinal weights:");
	print_vector(model->weights);

	printf("\nfinal bias: %.2f\n", model->bias);

	puts("\nfinal predictions:");
	print_vector(predictions);

	puts("\ntargets:");
	print_vector(y);

	printf("\ninitial loss: %.2f\n", init_loss);
	printf("final loss: %.2f (after %d iterations)\n", loss, iters);

    /*** free allocated data ***/
	free_matrix(X);
	free_vector(predictions); free_vector(y); free_vector(powers);
	polreg_free(model);
}