import numpy as np
# Polynomial integral
def polynomial_integral(k, a, b):
    return (b**(k+1) - a**(k+1)) / (k + 1)

# Fermi-Dirac integral
def fermi_dirac_integral(k, a, b):
    return (np.log(np.exp(k * b) + 1) - np.log(np.exp(k * a) + 1)) / k

# Example usage
k = 2
a, b = 0, 1
print("Polynomial Integral:", polynomial_integral(k, a, b))
print("Fermi-Dirac Integral:", fermi_dirac_integral(k, a, b))




