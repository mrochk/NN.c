#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "../vector/vector.h"
#include "../matrix/matrix.h"

/* calculate the dot product between two vectors */
float dotprod(Vector v, Vector w) {
    assert(v->n == w->n);

    int N = v->n;
    float dot = 0.0f;
    for (int i = 0; i < N; i++) {
        dot += v->d[i] * w->d[i];
    }

    return dot;
}
