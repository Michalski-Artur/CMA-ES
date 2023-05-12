import numpy as np
from vector_initializers.vector_initializer import VectorInitializer


class ZeroInitializer(VectorInitializer):

    def __init__(self):
        super().__init__()

    def initialize_vector(self, length, pop_size):
        return np.zeros(length)
