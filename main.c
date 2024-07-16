#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "tensors/tensors.h"
#include "utils/utils.h"
#include "linreg/linreg.h"

int main(int argc, char** argv) {
    srand(time(NULL));

    Vector x = new_zeros_vector(3);
    x->data[0] = 1.f;
    x->data[1] = 2.f;
    x->data[2] = 3.f;

    Linreg model = new_linreg(0.01f, 3);
    float pred = linreg_predict(model, x);
    printf("%.2f\n", pred);

    free_linreg(model);
    free_vector(x);

    return 0;
}