#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* Random float between zero and one. */
float randfloat() {
    float dividend = (float)rand();
    float divisor  = (float)RAND_MAX;
    return dividend / divisor;
}

/* Random integer casted to float in range [0, high]. */
float randint(int high) {
    return (float) (rand() % (high + 1));
}

/* max(a, b) */
float max(float a, float b) {
    return (a > b) ? a : b;
}
