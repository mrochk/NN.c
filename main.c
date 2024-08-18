#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//#include "examples/linreg/example.c"
//#include "examples/polreg/example.c"
//#include "examples/neurons/example.c"
#include "examples/nn/example.c"

#define ITERS_DEFAULT 200

int main(int argc, char** argv) {
    srand(time(NULL)); /* set seed here if needed */

    // linreg_run_eg(ITERS_DEFAULT);
    // polreg_run_eg(ITERS_DEFAULT);
    // neurons_run_eg(ITERS_DEFAULT);

    neuralnet_run_eg(ITERS_DEFAULT);

    return EXIT_SUCCESS;
}
