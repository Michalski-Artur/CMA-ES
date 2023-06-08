import numpy as np

from cma import CMA
from vector_initializer import XavierInitializer
from fitness_functions import quadratic, rastrigin

def main():
    seed = 30
    n = 2
    sigma = 1.3
    mean = np.zeros(n)

    initializer = XavierInitializer(seed)
    optimizer = CMA(mean=mean, sigma=sigma, vector_initializer=initializer)

    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = quadratic(x[0], x[1])
            solutions.append((x, value))
            print(f"{optimizer.generation:3d}  {value:10.5f}  " + "  ".join(f"{x_i}:6.2" for x_i in x))
        optimizer.tell(solutions)

        if optimizer.should_stop():
            break


if __name__ == "__main__":
    main()
