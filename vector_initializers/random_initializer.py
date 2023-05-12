from numpy import random
from vector_initializers.vector_initializer import VectorInitializer


class RandomInitializer(VectorInitializer):

    def __init__(self, seed):
        super().__init__(seed)

    def initialize_vector(self, length, pop_size):
        return random.rand(length)
