from abc import ABC, abstractmethod
from math import sqrt

from numpy import random
import numpy as np


class VectorInitializer(ABC):
    """Interface for initializing vector based on provided length and population size."""

    def __init__(self, seed=0):
        self.seed = seed
        random.seed(seed)

    @abstractmethod
    def initialize_vector(self, length, pop_size):  # Should it be normalized?
        pass


class XavierInitializer(VectorInitializer):
    """Initialize each element of the vector with value from U [-(1/sqrt(n)), 1/sqrt(n)] distribution."""

    def initialize_vector(self, length, pop_size):
        return random.uniform(-1.0/sqrt(pop_size), 1.0/sqrt(pop_size), length)


class HeInitializer(VectorInitializer):
    """Initialize each element of the vector with value from N (0.0, sqrt(2/n)) distribution."""

    def initialize_vector(self, length, pop_size):
        return random.normal(0, sqrt(2.0/pop_size), length)


class RandomInitializer(VectorInitializer):
    """Initialize each element of the vector with random value."""

    def initialize_vector(self, length, pop_size):
        return random.rand(length)


class ConstInitializer(VectorInitializer):
    """Initialize each element of the vector with constant value specified as parameter of the constructor."""
    def __init__(self, seed, value):
        super().__init__(seed)
        self.value = value

    def initialize_vector(self, length, pop_size):
        return np.full(length, self.value)


class ZeroInitializer(VectorInitializer):
    """Initialize each element of the vector with zero"""

    def initialize_vector(self, length, pop_size):
        return np.zeros(length)


class ReverseSquareRootInitializer(VectorInitializer):
    """Initialize each element of the vector with reverse square of population size"""

    def initialize_vector(self, length, pop_size):
        return np.full(length, 1/sqrt(pop_size))