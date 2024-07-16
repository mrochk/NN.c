#include "linreg.h"
#include "../utils/utils.h"
#include "../tensors/tensors.h"

Linreg new_linreg(float lr, int n_features) {
    Linreg linreg = (Linreg) malloc(sizeof(Linreg));
    linreg->lr = lr;
    linreg->weights = new_random_vector(n_features);
    linreg->bias = random_float();

    return linreg;
}

void free_linreg(Linreg linreg) {
    free_vector(linreg->weights);
    free(linreg);
}

float linreg_predict(Linreg linreg, Vector x) {
    return dotprod(linreg->weights, x) + linreg->bias;
}