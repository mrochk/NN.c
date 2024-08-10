#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#include "../../loss/loss.h"
#include "../../utils/utils.h"
#include "../../tensors/tensors.h"

#include "../../models/neuron/neuron.h"

Matrix forward(Neuron nh1, Neuron nh2, Neuron no, Matrix X, Vector preds) {
    Vector h1 = vector_new_(preds->n);
    h1 = neuron_forward_batch(nh1, X, h1);

    Vector h2 = vector_new_(preds->n);
    h2 = neuron_forward_batch(nh2, X, h2);

    Vector vectors[] = {h1, h2};
    Matrix H = matrix_from_vectors(2, vectors, 1);

    vector_free(h1); vector_free(h2);

    preds = neuron_forward_batch(no, H, preds);

    return H;
}

void neurons_run_eg(int iters) {
    /*** create synthetic data using f(x, y) = sin(x) + 0.1y^2 ***/
    const int n_features = 2;
    const int datapoints = 20;

	/* features matrix */
    Matrix X = matrix_new_randint_(datapoints, n_features, 20); 

	/* targets */
	Vector y = vector_new_(datapoints);
	for (int i = 0; i < y->n; i++) {
		Vector row = X->d[i];
		float x1 = row->d[0], x2 = row->d[1];
		y->d[i] = sinf(x1) + .1F*(x2*x2);
	}

	puts("features:");
	matrix_print(X);

	puts("\ntargets:");
	vector_print(y);

    /*** initialize the neurons ***/
    Neuron nh1 = neuron_new_(n_features, ReLU); /* hidden neuron 1 */
    Neuron nh2 = neuron_new_(n_features, ReLU); /* hidden neuron 2 */
    Neuron no  = neuron_new_(2, Linear);        /* output neuron   */

    /*** perform initial predictions ***/
	Vector predictions = vector_new_zeros_(datapoints);

	Matrix H = forward(nh1, nh2, no, X, predictions);
	matrix_free(H);

	puts("\ninitial predictions:");
	vector_print(predictions);

	float loss = MSE(predictions, y);
	printf("\ninitial loss (mse) : %.2f\n", loss);

    ///*** training loop ***/
	float lr = 0.001; /* learning rate */

	Vector dw_h1 = vector_new_zeros_(nh1->w->n); /* will store ∇_{w_h1,1, w_h1,2}L */
	Vector dw_h2 = vector_new_zeros_(nh1->w->n); /* will store ∇_{w_h2,1, w_h2,2}L */
	Vector dw_o  = vector_new_zeros_(no->w->n);  /* will store ∇_{w_o1, w_o2}L     */

	/* ∇_{b_h1, b_h2, b_o}L */
	float db_h1 = 0.F, db_h2 = 0.F, db_o = 0.F;

	for (int iter = 0; iter < 100; iter++) {
		/* set gradients to zero */
		dw_h1 = vector_set_zeros(dw_h1);
		dw_h2 = vector_set_zeros(dw_h2);
		dw_o  = vector_set_zeros(dw_o);
		db_h1 = db_h2 = db_o = 0.F;

		/* forward */
		H = forward(nh1, nh2, no, X, predictions);

		loss = MSE(predictions, y); 
		printf("-- iter %d, loss: %.2f\n", iter+1, loss);

		/* backward */

		/* compute dL/dw_o */
		for (int i = 0; i < datapoints; i++) {
			float error_i = y->d[i] - predictions->d[i];
			float h_i1 = H->d[i]->d[0];
			float h_i2 = H->d[i]->d[1];

			// dL/dwo_j = mean sum of: [-2 * error_i * h_i,j]
			dw_o->d[0] += -2.F * error_i * h_i1;
			dw_o->d[1] += -2.F * error_i * h_i2;
		}
		dw_o->d[0] /= datapoints;
		dw_o->d[1] /= datapoints;
		
		/* compute dL/db_o */
		for (int i = 0; i < datapoints; i++) {
			float error_i = y->d[i] - predictions->d[i];
			// dL/db_o = mean sum of: [-2 * error_i]
			db_o += -2.F * error_i;
		}
		db_o /= datapoints;

		/* compute dL/dw_h */

		/* compute dL/db_h */

		/* update neurons */
		for (int i = 0; i < dw_o->n; i++) {
			no->w->d[i] -= lr * dw_o->d[i];
		}
		no->b -= lr * db_o;

		matrix_free(H);
	}

	/*** log results ***/

	puts("\nfinal preds:");
	vector_print(predictions);

	puts("targets:");
	vector_print(y);

	printf("\nfinal loss: %.2f\n", loss);

	puts("final weights (output neuron):");
	vector_print(no->w);

	printf("\nfinal bias (output neuron): %.2f\n", no->b);

    /*** free allocated data ***/
	matrix_free(X);
	vector_free(predictions); vector_free(y);
	vector_free(dw_h1); vector_free(dw_h2); vector_free(dw_o);
    neuron_free(nh1); neuron_free(nh2); neuron_free(no);
}