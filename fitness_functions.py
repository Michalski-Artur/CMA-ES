
import numpy as np 


def quadratic(x, coefficients=[1, 2, 3]):
    coefficients = np.array(coefficients)
    return np.sum(coefficients * np.power(x, 2))

def rastrigin(x, A=10):
    return A * len(x) + np.sum(np.power(x, 2) - A * np.cos(2 * np.pi * x))