# Feedforward Neural Network Example

Here, a small neural network attempts to learn $f(x, y, z) = \sin(\pi x) + \sin(2y) - z$. 

During backprop, gradients are computed via an approximation of the symmetric derivative $\frac{f(x+h) - f(x-h)}{2h}$. 

In the code, $h$ = `eps` = $1\mathrm{e}{-6}$.