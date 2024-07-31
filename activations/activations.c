#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../utils/utils.h"

#include "activations.h"

float relu(float x) {
    return max(x, 0.0f);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}