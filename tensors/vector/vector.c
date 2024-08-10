#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "vector.h"
#include "../../utils/utils.h"

Vector vector_new_(int n) {
    Vector vector = (Vector) malloc(sizeof(Vector));
    vector->n = n;
    vector->d = malloc(sizeof(float) * n);

    return vector;
}

Vector vector_new_zeros_(int n) {
    Vector vector = vector_new_(n);
    for (int i = 0; i < n; i++) { vector->d[i] = 0.0f; }

    return vector;
}

Vector vector_new_randfloat_(int n) {
    Vector vector = vector_new_(n);
    for (int i = 0; i < n; i++) { 
        vector->d[i] = randfloat(); 
    }

    return vector;
}

Vector vector_new_randint_(int n, int high) {
    Vector vector = vector_new_(n);
    for (int i = 0; i < n; i++) { 
        vector->d[i] = randint(high); 
    }

    return vector;
}

void vector_free(Vector v) {
    free(v->d); v->d = NULL;
    free(v); v = NULL;
}

Vector vector_set_zeros(Vector v) {
    for (int i = 0; i < v->n; i++) { v->d[i] = 0.F; }
    return v;
}

void vector_print(Vector v) {
    printf("[");
    for (int i = 0; i < v->n; i++) {
        float x = v->d[i];

        if (i == v->n-1) { 
            if (ceilf(x) == x) { printf("%d", (int)x); } 
            else { printf("%.2f", v->d[i]); }
        } 
        else { 
            if (ceilf(x) == x) { printf("%d ", (int)x); } 
            else { printf("%.2f ", v->d[i]); }
        }
    }
    puts("]");
}