import numpy as np
from vector_initializers.vector_initializer import VectorInitializer


class ConstInitializer(VectorInitializer):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def initialize_vector(self, length, pop_size):
        return np.full(length, self.value)
