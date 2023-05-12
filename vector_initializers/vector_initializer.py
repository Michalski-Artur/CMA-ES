from abc import ABC, abstractmethod
from math import sqrt

from numpy import random
import numpy as np


class VectorInitializer(ABC):
    """VectorInitializer

        Interface for initializing vector based on provided length and population size.
    """

    def __init__(self, seed=0):
        self.seed = seed
        random.seed(seed)

    @abstractmethod
    def initialize_vector(self, length, pop_size):  # Should it be normalized?
        pass


class XavierInitializer(VectorInitializer):
    """XavierInitializer

        Initialize each element of the vector with value from U [-(1/sqrt(n)), 1/sqrt(n)] distribution
    """

    def __init__(self, seed):
        super().__init__(seed)

    def initialize_vector(self, length, pop_size):
        return random.uniform(-1.0/sqrt(pop_size), 1.0/sqrt(pop_size), length)


class HeInitializer(VectorInitializer):
    """HeInitializer

        Initialize each element of the vector with value from N (0.0, sqrt(2/n)) distribution
    """

    def __init__(self, seed):
        super().__init__(seed)

    def initialize_vector(self, length, pop_size):
        return random.normal(0, sqrt(2.0/pop_size), length)


class RandomInitializer(VectorInitializer):
    """RandomInitializer

        Initialize each element of the vector with random value
    """
    def __init__(self, seed):
        super().__init__(seed)

    def initialize_vector(self, length, pop_size):
        return random.rand(length)


class ConstInitializer(VectorInitializer):
    """ConstInitializer

        Initialize each element of the vector with constant value specified as parameter of the constructor
    """
    def __init__(self, value):
        super().__init__()
        self.value = value

    def initialize_vector(self, length, pop_size):
        return np.full(length, self.value)


class ZeroInitializer(VectorInitializer):
    """RandomInitializer

        Initialize each element of the vector with zero
    """
    def __init__(self):
        super().__init__()

    def initialize_vector(self, length, pop_size):
        return np.zeros(length)
