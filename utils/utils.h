#ifndef UTILS_H
#define UTILS_H

#include <math.h>

#define MAX_ITERS_LOG 100

#define PI acosf(-1.0F)

typedef unsigned int uint;

typedef struct { int a, b; } Pair;

Pair pair(int a, int b);

float randfloat();

float randint(int high);

float maxf(float a, float b);

float absf(float x);

#endif

