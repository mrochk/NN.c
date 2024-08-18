#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "../../utils/utils.h"
#include "../matrix/matrix.h"

#include "vector.h"

Vector vector_new_(uint n) {
    Vector vector = (Vector) malloc(sizeof(struct Vector_t));
    vector->d = malloc(sizeof(float) * n);
    vector->n = n;

    return vector;
}

Vector vector_new_zeros_(uint n) {
    Vector vector = vector_new_(n);
    vector_set_zeros(vector);

    return vector;
}

Vector vector_new_randfloat_(uint n) {
    Vector vector = vector_new_(n);
    for (int i = 0; i < n; i++) { vector->d[i] = randfloat(); }

    return vector;
}

Vector vector_new_randint_(uint n, int high) {
    Vector vector = vector_new_(n);
    for (int i = 0; i < n; i++) { vector->d[i] = randint(high); }

    return vector;
}

Vector vector_new_from_(Vector v) {
    assert(v);

    Vector vector = vector_new_(v->n);
    vector_copy(vector, v);

    return vector;
}

void vector_free(Vector v) {
    assert(v);

    free(v->d);
    free(v); 

    return;
}

/* compute v := w */
void vector_copy(Vector v, Vector w) {
    assert(v); assert(w); assert(v->n == w->n);

    for (int i = 0; i < v->n; i++) { v->d[i] = w->d[i]; }

    return;
}

void vector_set_zeros(Vector v) {
    assert(v);

    for (int i = 0; i < v->n; i++) { v->d[i] = 0.F; }

    return;
}

void vector_print(Vector v) {
    assert(v);

    printf("[");
    for (int i = 0; i < v->n; i++) {
        float x = v->d[i];
        if (i == v->n-1) { 
            if (ceilf(x) == x) { printf("%d", (int)x); } 
            else { printf("%.2f", v->d[i]); }
            continue;
        } 
        if (ceilf(x) == x) { printf("%d ", (int)x); } 
        else { printf("%.2f ", v->d[i]); }
    }
    puts("]");

    return;
}