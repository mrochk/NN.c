# nn.c
Multi Layer Perceptron & Linear Regression implementation from scratch in pure C.

Models implemented:
- (multivariate) Linear regression. [\[linreg\]](models/linreg) [\[example\]](examples/linreg)
- (multivariate) Polynomial regression. [\[polreg\]](models/polreg) [\[example\]](examples/polreg)
- Multilayer Perceptron at the neuron (perceptron) level. [\[neuron\]](models/neuron) [\[example\]](examples/neurons)
<!-- - Multilayer Perceptron. [\[mlp\]](models/mlp) -->

You can find example of using these models in [examples](examples).

*Any function allocating memory on the heap ends with an underscore '_'.*\
*The pointers returned by these must be freed by their associated functions.*

For now the functions derivatives are derived by hand for each example but I plan on building an autograd engine soon too.

For the optimizer we always use Stochastic Gradient Descent.
