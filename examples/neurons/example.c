#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#include "../../loss/loss.h"
#include "../../utils/utils.h"
#include "../../tensors/tensors.h"

#include "../../models/neuron/neuron.h"

Vector forward(Neuron nh1, Neuron nh2, Neuron no, Matrix X, Vector preds) {
    Vector h1 = vector_new_(preds->n);
    h1 = neuron_forward_batch(nh1, X, h1);

    Vector h2 = vector_new_(preds->n);
    h2 = neuron_forward_batch(nh2, X, h2);

    Vector vectors[] = {h1, h2};
    Matrix H = matrix_from_vectors(2, vectors, 1);

    vector_free(h1); vector_free(h2);

    preds = neuron_forward_batch(no, H, preds);

    matrix_free(H);

    return preds;
}

void neurons_run_eg(int iters) {
    /*** create synthetic data using f(x, y) = sin(x) + 0.1y^2 ***/
    const int n_features = 2;
    const int datapoints = 20;

	/* features */
    Matrix X = matrix_new_randint_(datapoints, n_features, 20); /* features matrix */

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
    Neuron nh1 = neuron_new_(2, ReLU);   /* hidden neuron 1 */
    Neuron nh2 = neuron_new_(2, ReLU);   /* hidden neuron 2 */
    Neuron no  = neuron_new_(2, Linear); /* output neuron   */

    /*** perform initial predictions ***/
	Vector predictions = vector_new_zeros_(datapoints);

	predictions = forward(nh1, nh2, no, X, predictions);

	puts("\ninitial predictions:");
	vector_print(predictions);

	float loss = MSE(predictions, y);
	printf("\ninitial loss (mse) : %.2f\n", loss);

    ///*** training loop ***/
	//Vector dw = vector_new_zeros_(model->weights->n);
	//float  db = 0.0F;

	//for (int i = 0; i < iters; i++) {
		///*** forward ***/
		//predictions = linreg_predict_batch(model, X, predictions);

		//loss = MAE(predictions, y);

        //if (iters < MAX_ITERS_LOG) {
		    //printf("iter %d, loss: %.2f\n", i+1, loss);
        //}

		///*** backward ***/
		//db = 0.0F;
		//for (int k = 0; k < dw->n; k++) { dw->d[k] = 0.0F; }

		///* calculating dw and db */
		//for (int j = 0; j < datapoints; j++) {
			//float e    = y->d[j] - predictions->d[j];
			//float dabs = e / absf(e);

			//db += dabs;

			//for (int k = 0; k < dw->n; k++) {
				//dw->d[k] += -dabs * X->d[j]->d[k];
			//}
		//}

		///* updating weights */
		//model->bias -= model->lr * db;

		//for (int k = 0; k < dw->n; k++) { 
			//dw->d[k] /= (float)datapoints; 
			//model->weights->d[k] -= model->lr * dw->d[k];
		//}
	//}

	//puts("\nfinal weights:");
	//vector_print(model->weights);

	//printf("\nfinal bias: %.2f\n", model->bias);

	//puts("\nfinal predictions:");
	//vector_print(predictions);

	//puts("\ntargets:");
	//vector_print(y);

	//printf("\nfinal loss: %.2f\n", loss);

    /*** free allocated data ***/
	matrix_free(X);
	vector_free(predictions); vector_free(y); // vector_free(dw);
    neuron_free(nh1); neuron_free(nh2); neuron_free(no);
}