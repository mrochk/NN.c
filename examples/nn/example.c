#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "../../loss/loss.h"
#include "../../utils/utils.h"
#include "../../tensors/tensors.h"

#include "../../models/nn/nn.h"

void neuralnet_run_eg(int iters) {
    /* create data */
    Vector x = vector_new_randint_(5, 10);

    puts("x:"); vector_print(x);

    Pair l1 = pair(5, 3);
    Pair l2 = pair(3, 1);
    Pair arch[] = {pair(5, 3), pair(3, 1)};

    NN nn = nn_new_(2, arch, Identity);

    Vector o = vector_new_(1);

    nn_forward(nn, x, o);

    puts("o:"); vector_print(o);

    vector_free(x); vector_free(o);
    nn_free(nn);
}