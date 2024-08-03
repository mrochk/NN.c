#include <stdlib.h>
#include <stdio.h>

#include "vector.h"
#include "../../utils/utils.h"

Vector new_zeros_vector_(int n) {
    Vector vector = (Vector)malloc(sizeof(Vector));

    vector->n = n;

    vector->d = malloc(sizeof(float) * n);

    for (int i = 0; i < n; i++) { 
        vector->d[i] = 0.0f; 
    }

    return vector;
}

Vector new_random_float_vector_(int n) {
    Vector vector = (Vector)malloc(sizeof(Vector));

    vector->n = n;

    vector->d = malloc(sizeof(float) * n);

    for (int i = 0; i < n; i++) { 
        vector->d[i] = randfloat(); 
    }

    return vector;
}

Vector new_random_int_vector_(int n, int high) {
    Vector vector = (Vector)malloc(sizeof(Vector));

    vector->n = n;

    vector->d = malloc(sizeof(float) * n);

    for (int i = 0; i < n; i++) { 
        vector->d[i] = randint(high); 
    }

    return vector;
}

void free_vector(Vector v) {
    free(v->d);
    free(v);
}

void print_vector(Vector v) {
    printf("[");
    for (int i = 0; i < v->n; i++) {
        if (i == v->n-1) { printf("%04.1f", v->d[i]); } 
        else { printf("%04.1f ", v->d[i]); }
    }
    puts("]");
}