#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "neuron/neuron.h"
#include "activations/activations.h"
#include "utils/utils.h"

#include "examples/eg_linreg.c"
#include "examples/eg_neurons.c"

int main(int argc, char** argv) {
    srand(time(NULL));

    //run_linreg_example(100);

    run_neurons_example();

    return EXIT_SUCCESS;
}