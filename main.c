#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utils/utils.h"
#include "tensors/tensors.h"
#include "activations/activations.h"

#include "examples/linreg/example.c"
#include "examples/polreg/example.c"
#include "examples/neurons/example.c"

#include "models/nn/nn.h"

#define ITERS_DEFAULT 50

int main(int argc, char** argv) {
    srand(time(NULL));

    //linreg_run_eg(ITERS_DEFAULT);
    //polreg_run_eg(ITERS_DEFAULT);
    //neurons_run_eg(ITERS_DEFAULT);

    Matrix X = matrix_new_randint_(1000, 5, 10);

    Pair l1 = {5U,  10U};
    Pair l2 = {10U, 10U};
    Pair l3 = {10U,  3U};

    Pair arch[] = {l1, l2, l3};

    Matrix Out = matrix_new_(1000, 3);

    NN nn = nn_new_(3, arch, ReLU);

    matrix_print(X);

    nn_forward_batch(nn, X, Out);

    matrix_apply(Out, Sigmoid);

    matrix_print(Out);

    matrix_free(Out); matrix_free(X);
    nn_free(nn);

    return EXIT_SUCCESS;
}
