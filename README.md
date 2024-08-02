# nn.c
Multi Layer Perceptron & Linear Regression implementation from scratch in pure C.

I implemented:
- The (multivariate) linear regression. [\[linreg\]](models/linreg)
- The (multivariate) polynomial regression. [\[polreg\]](models/polreg)
- The neural network neuron. [\[neuron\]](models/neuron)
<!-- - A complete multi-layer perceptron. -->

You can find example of using these models in [examples](examples).

*Any function allocating memory on the heap ends with an underscore '_'.*\
*The pointers returned by these must be freed by their associated functions.*

For now the functions derivatives are derived by hand for each example but I plan on building an autograd engine soon too.

For the optimizer we always use Stochastic Gradient Descent.
