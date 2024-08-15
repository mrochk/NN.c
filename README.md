# NN.c
Feedforward neural network & other ML models implemented from scratch in pure C.

Models implemented:
- (multivariate) Linear regression. [\[linreg\]](models/linreg) [\[example\]](examples/linreg)
- (multivariate) Polynomial regression. [\[polreg\]](models/polreg) [\[example\]](examples/polreg)
- Artificial neuron (perceptron). [\[neuron\]](models/neuron) [\[example\]](examples/neurons)
- Feedforward neural network. [\[nn\]](models/nn) [\[example\]](examples/nn)

For now the functions derivatives / weights gradients are calculated *"by hand"* for each example but I plan on building an autograd engine soon too.

For the optimizer we always use the simple *non-stochastic* gradient descent algorithm.

For storing numerical values (parameters etc) we use the single-precision floating-point type (`float`).
