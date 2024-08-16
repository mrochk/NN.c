# Feedforward Neural Network Example

Here, a small neural network made of 3 neurons attempts to learn $f(x, y, z) = \sin(x) + \sin(2y) + z$. 

Gradients are computed via an approximation of the symmetric derivative $\frac{f(x+h) - f(x-h)}{2h}$, in the code `eps` = $h$ = $1\mathrm{e}{-6}$.