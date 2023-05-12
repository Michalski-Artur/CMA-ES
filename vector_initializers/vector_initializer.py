from abc import ABC, abstractmethod
from numpy import random


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
