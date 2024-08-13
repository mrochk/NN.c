#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "utils.h"

/* create a new pair of ints */
Pair pair(int a, int b) { Pair p = {a, b}; return p; }

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

/* max(float a, float b) */
float maxf(float a, float b) {
    return (a > b) ? a : b;
}

/* abs(float x) */
float absf(float x) {
    return (x < 0.0F) ? -x : x;
}