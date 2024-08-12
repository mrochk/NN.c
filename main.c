#include <stdlib.h>
#include <stdio.h>
#include <math.h>

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

    Vector out = vector_new_zeros_(3);

    Layer layer = layer_new_(2, 3, ReLU);

    Vector x = vector_new_randint_(2, 10);
    x->d[0] = -4.F;

    puts("out:");
    vector_print(out);

    puts("x:");
    vector_print(x);

    puts("W:");
    matrix_print(layer->weights);

    puts("b:");
    vector_print(layer->biases);

    layer_forward(layer, x, out);

    puts("out:");
    vector_print(out);

    return EXIT_SUCCESS;
}
