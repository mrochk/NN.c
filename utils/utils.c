#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "utils.h"

/* create a new pair of uints */
UintPair pair(uint a, uint b) { return (UintPair){a, b}; }

/* returns a random float in [-1, 1] */
float randfloat() {
    float a = (float) rand();
    float b = (float) RAND_MAX ;
    return a / b * 2.F - 1.F;
    /* given by claude */
}

/* returns a random integer casted to float, in [0, high] */
float randint(int high) { return (float) (rand() % (high + 1)); }

/* max function for floats */
float maxf(float a, float b) { return (a > b) ? a : b; }

/* abs function for floats */
float absf(float x) { return (x < 0.F) ? -x : x; }