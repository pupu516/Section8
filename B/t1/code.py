import numpy as np
class Quadrature:
    def midpoint(self, f, a, b):
        return (b - a) * f((a + b) / 2)

    def trapezoidal(self, f, a, b):
        return (b - a) / 2 * (f(a) + f(b))

    def simpson(self, f, a, b):
        return (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b))

# Example usage
quad = Quadrature()
f = lambda x: x**2
a, b = 0, 1
print("Midpoint:", quad.midpoint(f, a, b))
print("Trapezoidal:", quad.trapezoidal(f, a, b))
print("Simpson's:", quad.simpson(f, a, b))



from scipy.special import roots_legendre

class GaussQuad(Quadrature):
    def __init__(self, order):
        self.order = order
        self.roots, self.weights = roots_legendre(order)

    def integrate(self, f, a, b):
        # Transform roots from [-1, 1] to [a, b]
        x_transformed = (b - a) / 2 * self.roots + (a + b) / 2
        return (b - a) / 2 * np.sum(self.weights * f(x_transformed))

# Example usage
gauss = GaussQuad(order=5)
print("Gauss-Legendre:", gauss.integrate(f, a, b))





from scipy.special import legendre

def newton_method(poly, deriv, x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        x_new = x - poly(x) / deriv(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

# Example: Find roots of Legendre polynomial of order 3
M = 3
poly = legendre(M)
deriv = poly.deriv()
roots = [newton_method(poly, deriv, x0) for x0 in np.linspace(-1, 1, M)]
print("Roots:", roots)




