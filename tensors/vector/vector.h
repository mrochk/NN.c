#ifndef VECTOR_H
#define VECTOR_H

/* vector of floats of dimension n */
struct Vector_t {
    /* vector dimension */
    size_t n;
    /* vector data */
    float* d;
};

/* vector of floats of dimension n */
typedef struct Vector_t* Vector;

/* create a new vector of dimension n not initialized with any value */
Vector vector_new_(int n);

/* create a new vector of dimension n filled with zeros */
Vector vector_new_zeros_(int n);

/* create a new vector of dimension n filled with random floats in [0, 1] */
Vector vector_new_randfloat_(int n);

/* create a new vector of dimension n filled with random ints in [0, high] */
Vector vector_new_randint_(int n, int high);

/* free a vector allocated with new_..._vector_ */
void vector_free(Vector v);

/* set all values of v to 0 */
void vector_set_zeros(Vector v);

/* print a vector */
void vector_print(Vector v);

#endif