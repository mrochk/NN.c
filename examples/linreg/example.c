#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "../../tensors/tensors.h"
#include "../../utils/utils.h"
#include "../../loss/loss.h"

#include "../../models/linreg/linreg.h"

void linreg_run_eg(int iters) {
    /*** create synthetic data using f(x, y, z) = 1x + 2y + 3z - 4 ***/
    const int n_features = 3;
    const int datapoints = 20;

	/* features */
    Matrix X = new_random_int_matrix_(datapoints, n_features, 20); /* features matrix */

	/* targets */
	Vector y = new_zeros_vector_(datapoints);
	for (int i = 0; i < y->n; i++) {
		Vector row = X->d[i];
		float x1 = row->d[0], x2 = row->d[1], x3 = row->d[2];
		y->d[i] = x1 + 2.0F*x2 + 3.0F*x3 - 4.0F;
	}

	puts("features:");
	print_matrix(X);

	puts("\ntargets:");
	print_vector(y);

    /*** initialize the model ***/
	Linreg model = new_linreg(0.01F /* loss */, n_features );

    /*** perform initial predictions ***/
	Vector predictions = new_zeros_vector_(datapoints);

	predictions = linreg_predict_batch(model, X, predictions);

	puts("\ninitial predictions:");
	print_vector(predictions);

	float loss = MAE(predictions, y);
	printf("\ninitial loss (mae) : %.2f\n", loss);

    /*** training loop ***/
	Vector dw = new_zeros_vector_(model->weights->n);
	float  db = 0.0F;

	for (int i = 0; i < iters; i++) {
		/*** forward ***/
		predictions = linreg_predict_batch(model, X, predictions);

		loss = MAE(predictions, y);
		printf("iter %d, loss: %.2f\n", i, loss);

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
	print_vector(model->weights);

	printf("\nfinal bias: %.2f\n", model->bias);

	puts("\nfinal predictions:");
	print_vector(predictions);

	puts("\ntargets:");
	print_vector(y);

	printf("\nfinal loss: %.2f\n", loss);

    /*** free allocated data ***/
	free_matrix(X);
	free_vector(predictions); free_vector(y); free_vector(dw);
	free_linreg(model);
}