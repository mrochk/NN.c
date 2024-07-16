typedef struct {
    int m, n;
    float** data;
} Matrix_t;

typedef Matrix_t* Matrix;

Matrix new_zeros_matrix(int m, int n);

Matrix new_random_matrix(int m, int n);

void free_matrix(Matrix M);

void print_matrix(Matrix M);