import numpy as np

from cmaes.cma import CMA
from vector_initializers.vector_initializer import \
    XavierInitializer, HeInitializer, ZeroInitializer, ConstInitializer, RandomInitializer


def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2


def main():
    seed = 30
    initializer = XavierInitializer(seed)
    optimizer = CMA(mean=np.zeros(2), sigma=1.3, vector_initializer=initializer)
    print(" g    f(x1,x2)     x1      x2  ")
    print("===  ==========  ======  ======")

    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = quadratic(x[0], x[1])
            solutions.append((x, value))
            print(
                f"{optimizer.generation:3d}  {value:10.5f}"
                f"  {x[0]:6.2f}  {x[1]:6.2f}"
            )
        optimizer.tell(solutions)

        if optimizer.should_stop():
            break


if __name__ == "__main__":
    main()
