#include <stdlib.h>
#include <stdio.h>

#include "../../tensors/tensors.h"
#include "../../utils/utils.h"
#include "../../loss/loss.h"

#include "../../models/linreg/linreg.h"

void linreg_run_eg(int iters) {
    /*** create synthetic data using f(x, y, z) = 1x + 2y + 3z - 4 ***/
    const int n_features = 3;
    const int datapoints = 20;

	/* features */
    Matrix X = matrix_new_randint_(datapoints, n_features, 20); /* features matrix */

	/* targets */
	Vector y = vector_new_zeros_(datapoints);
	for (int i = 0; i < y->n; i++) {
		Vector row = X->d[i];
		float x1 = row->d[0], x2 = row->d[1], x3 = row->d[2];
		y->d[i] = x1 + 2.0F*x2 + 3.0F*x3 - 4.0F;
	}

	puts("features:");
	matrix_print(X);

	puts("\ntargets:");
	vector_print(y);

    /*** initialize the model ***/
	Linreg model = linreg_new_(0.01F /* loss */, n_features );

    /*** perform initial predictions ***/
	Vector predictions = vector_new_zeros_(datapoints);

	predictions = linreg_predict_batch(model, X, predictions);

	puts("\ninitial predictions:");
	vector_print(predictions);

	float loss = MAE(predictions, y);
	printf("\ninitial loss (mae) : %.2f\n", loss);

    /*** training loop ***/
	Vector dw = vector_new_zeros_(model->weights->n);
	float  db = 0.0F;

	for (int i = 0; i < iters; i++) {
		/*** forward ***/
		predictions = linreg_predict_batch(model, X, predictions);

		loss = MAE(predictions, y);

        if (iters < MAX_ITERS_LOG) {
		    printf("iter %d, loss: %.2f\n", i+1, loss);
        }

		/*** backward ***/
		db = 0.0F;
		for (int k = 0; k < dw->n; k++) { dw->d[k] = 0.0F; }

		/* calculating dw and db */
		for (int j = 0; j < datapoints; j++) {
			float e    = y->d[j] - predictions->d[j];
			float dabs = e / absf(e);

			db += dabs;

			for (int k = 0; k < dw->n; k++) {
				dw->d[k] += -dabs * X->d[j]->d[k];
			}
		}

		/* updating weights */
		model->bias -= model->lr * db;

		for (int k = 0; k < dw->n; k++) { 
			dw->d[k] /= (float)datapoints; 
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

	printf("\nfinal loss: %.2f\n", loss);

    /*** free allocated data ***/
	matrix_free(X);
	vector_free(predictions); vector_free(y); vector_free(dw);
	linreg_free(model);

	return;
}