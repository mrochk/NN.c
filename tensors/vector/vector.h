#ifndef VECTOR_H
#define VECTOR_H

/* vector of floats of dimension n */
typedef struct {
    /* vector dimension */
    int n;
    /* vector data */
    float* d;
} Vector_t;

/* vector of floats of dimension n */
typedef Vector_t* Vector;

/* create a new vector of dimension n filled with zeros */
Vector new_zeros_vector_(int n);

/* create a new vector of dimension n filled with random floats in [0, 1] */
Vector new_random_float_vector_(int n);

/* create a new vector of dimension n filled with random ints in [0, high] */
Vector new_random_int_vector_(int n, int high);

/* free a vector allocated with new_..._vector_ */
void free_vector(Vector v);

/* print a vector on one line */
void print_vector(Vector v);

#endif