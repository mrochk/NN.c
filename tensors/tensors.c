#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "tensors.h"
#include "matrix/matrix.h"
#include "vector/vector.h"

float dotprod(Vector v, Vector w) {
    assert(v->n == w->n);

    int n = v->n;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += v->data[i] * w->data[i]; 
    }

    return sum;
}