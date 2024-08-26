#include <stdlib.h>
#include <stdio.h>
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
    Matrix X = matrix_new_randfloat_(datapoints, n_features); /* features matrix */
    /* copying the first column in the second */
    for (int i = 0; i < X->m; i++) {
        X->d[i]->d[1] = X->d[i]->d[0];
        X->d[i]->d[2] = X->d[i]->d[0];
    }

	/* targets */
	Vector y = vector_new_zeros_(datapoints);
	for (int i = 0; i < y->n; i++) {
		float x = X->d[i]->d[0];
		y->d[i] = sinf(2.F * PI * x);
	}

	puts("features:");
	matrix_print(X);

	puts("\ntargets:");
	vector_print(y);

    /*** initialize the model ***/
    Vector powers = vector_new_zeros_(n_features); 
    powers->d[0] = 3.0F; /* we use a polynomial of degree 3 */
    powers->d[1] = 2.0F; powers->d[2] = 1.0F;

	Polreg model = polreg_new_(powers, .01F);

    /*** perform initial predictions ***/
	Vector predictions = vector_new_zeros_(datapoints);

	predictions = polreg_predict_batch(model, X, predictions);

	puts("\ninitial predictions:");
	vector_print(predictions);

	float init_loss = MSE(predictions, y);
	printf("\ninitial loss (mse) : %.2f\n", init_loss);

    /*** training loop ***/
	float  db = 0.F, loss;
    Vector dw = vector_new_zeros_(n_features);

	for (int i = 0; i < iters; i++) {
		/*** forward ***/
		predictions = polreg_predict_batch(model, X, predictions);

		loss = MSE(predictions, y);

        if (iters < MAX_ITERS_LOG) {
		    printf("iter %d, loss: %.2f\n", i+1, loss);
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
	vector_print(model->weights);

	printf("\nfinal bias: %.2f\n", model->bias);

	puts("\nfinal predictions:");
	vector_print(predictions);

	puts("\ntargets:");
	vector_print(y);

	printf("\ninitial loss: %.2f\n", init_loss);
	printf("final loss: %.2f (after %d iterations)\n", loss, iters);

    /*** free allocated data ***/
	matrix_free(X);
	vector_free(predictions); vector_free(y); vector_free(dw);
	polreg_free(model);
}