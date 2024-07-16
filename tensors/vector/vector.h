#ifndef VECTOR_H
#define VECTOR_H

typedef struct {
    int n;
    float* data;
} Vector_t;

typedef Vector_t* Vector;

Vector new_zeros_vector(int n);

Vector new_random_vector(int n);

void free_vector(Vector v);

void print_vector(Vector v);

#endif