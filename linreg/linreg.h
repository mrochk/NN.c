/*
A linear regression structure containing
a weight w and a bias b, to be fit to the data.
*/
typedef struct { 
    float w;
    float b;
} Linreg;

float init_param();

/*
Initialize a new linear regression model with randomly
initialized parameters.
*/
Linreg* new_linreg();

float* forward(Linreg* linreg, float* x, int n);

float squared_error(float pred, float target);

float grad_w(float* x, float* y, float* preds, int n); 

float grad_b(float* y, float* preds, int n);

float mse(float* preds, float* targets, int n);