#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "utils.h"

/* create a new pair of uints */
Pair pair(int a, int b) { return (Pair){a, b}; }

/* Random float in [-1, 1]. */
float randfloat() {
    float a = (float) rand();
    float b = (float) RAND_MAX ;
    return a / b * 2.F - 1.F;
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