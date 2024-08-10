# nn.c
Multilayer Perceptron & other ML models implemented from scratch in pure C.

Models implemented:
- (multivariate) Linear regression. [\[linreg\]](models/linreg) [\[example\]](examples/linreg)
- (multivariate) Polynomial regression. [\[polreg\]](models/polreg) [\[example\]](examples/polreg)
- Neural Network at the neuron level. [\[neuron\]](models/neuron) [\[example\]](examples/neurons)
<!-- - Multilayer Perceptron. [\[mlp\]](models/mlp) -->

For now the functions derivatives / weights gradients are calculated *"by hand"* for each example but I plan on building an autograd engine soon too.

For the optimizer we always use the simple *non-stochastic* gradient descent algorithm.

todo:
- [ ] finish neurons example
- [ ] implement mlp