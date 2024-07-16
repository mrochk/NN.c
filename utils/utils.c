#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* Random float between zero and one. */
float random_float() {
    float dividend = (float)rand();
    float divisor  = (float)RAND_MAX;
    return dividend / divisor;
}