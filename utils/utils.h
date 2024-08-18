#ifndef UTILS_H
#define UTILS_H

#include <math.h>

/* max number of iters before not printing the loss at each iteration */
#define MAX_ITERS_LOG 100

/* the pi constant */
#define PI acosf(-1.0F)

/* unsigned int */
typedef unsigned int uint;

/* a pair of unsigned integers */
typedef struct { uint a, b; } UintPair;

/* create a new pair of uints */
UintPair pair(uint a, uint b);

/* returns a random float in [-1, 1] */
float randfloat();

/* returns a random integer casted to float, in [0, high] */
float randint(int high);

/* max function for floats */
float maxf(float a, float b);

/* max function for float */
float absf(float x);

#endif

