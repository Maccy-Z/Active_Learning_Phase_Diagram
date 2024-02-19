from scipy.integrate import quad
import math


# Define the function you want to integrate
def my_function(x):
    g = 1 / (math.exp(math.cos(x)) + 2)
    f = 2.0775 * math.exp(math.cos(x)) / (1 + 2 * math.exp(math.cos(x))) + g
    return f


# Integrate the function from a to b (e.g., from 0 to 1)
integral, error = quad(my_function, 0, math.pi * 2)

print("Integral:", integral)
print(math.pi * 2)
print(integral / math.pi / 2)
