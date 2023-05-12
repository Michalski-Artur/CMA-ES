from math import sqrt
from numpy import random
from vector_initializers.vector_initializer import VectorInitializer


class HeInitializer(VectorInitializer):

    def __init__(self, seed):
        super().__init__(seed)

    def initialize_vector(self, length, pop_size):
        return random.normal(0, sqrt(2.0/pop_size), length)
