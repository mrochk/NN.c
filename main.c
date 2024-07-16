#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "tensors/tensors.h"
#include "utils/utils.h"

int main(int argc, char** argv) {
    srand(time(NULL));

    Matrix A = new_zeros_matrix(4, 3);
    Matrix B = new_random_matrix(3, 4);

    Vector v = new_random_vector(10);

    print_matrix(A);
    puts("");
    print_matrix(B);
    puts("");
    print_vector(v);

    free_matrix(A);
    free_matrix(B);
    free_vector(v);

    return 0;
}