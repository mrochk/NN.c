# nn.c
Multi Layer Perceptron & Linear Regression implementation from scratch in pure C.

I implemented:
- The (multivariate) linear regression. [linreg](linreg)
<!-- - The (multivariate) polynomial regression. [polreg](polreg) -->
- A simple neural network at the neuron level.
<!-- - A complete multi-layer perceptron. -->

*Any function allocating memory on the heap ends with an underscore '_'.*\
*The pointers returned by these must be freed by their associated functions.*

For now the functions derivatives are derived by hand but I plan on building an autograd engine soon too.

For the optimizer, for now we always use SGD.
