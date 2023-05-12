from math import sqrt
from numpy import random
from vector_initializers.vector_initializer import VectorInitializer


class XavierInitializer(VectorInitializer):

    def __init__(self, seed):
        super().__init__(seed)

    def initialize_vector(self, length, pop_size):
        return random.uniform(-1.0/sqrt(pop_size), 1.0/sqrt(pop_size), length)
