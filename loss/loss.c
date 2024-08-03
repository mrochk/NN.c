#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "../tensors/tensors.h"
#include "../utils/utils.h"

/* mean absolute error loss */
float MAE(Vector p, Vector t) {
    assert(p && t && p->n == t->n);

    float sum = 0.0f;
    for (int i = 0; i < p->n; i++) { 
        sum += absf(t->d[i] - p->d[i]); 
    }

    return sum / p->n;
}

/* mean squared error loss */
float MSE(Vector p, Vector t) {
    assert(p && t && p->n == t->n);

    float sum = 0.0f;
    for (int i = 0; i < p->n; i++) { 
        sum += powf(t->d[i] - p->d[i], 2.0F); 
    }

    return sum / p->n;
}