#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "../vector/vector.h"
#include "../matrix/matrix.h"

float dotprod(Vector v, Vector w) {
    assert(v->n == w->n);

    int N = v->n;
    float dot = 0.0f;
    for (int i = 0; i < N; i++) {
        dot += v->data[i] * w->data[i];
    }

    return dot;
}
