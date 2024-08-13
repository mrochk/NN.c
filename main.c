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
    //neurons_run_eg(10000);

    Pair l1 = {1, 2};
    Pair l2 = {2, 1};

    Pair nn_struct[] = {l1, l2};

    NN nn = nn_new_(2, nn_struct, ReLU);

    Vector x = vector_new_(1);
    x->d[0] = -10;

    vector_print(x);
    x = nn_forward(nn, x);

    vector_print(x);

    nn_free(nn);

    vector_free(x);

    return EXIT_SUCCESS;
}
