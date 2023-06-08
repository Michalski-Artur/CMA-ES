from cma import CMA
from vector_initializer import ReverseSquareRootInitializer, RandomInitializer

def _loop_cma_es(problem, es):
    while not es.should_stop():
        # Generate new candidate solutions using the ask() method
        solutions = [es.ask() for _ in range(es.population_size)]

        # Evaluate the objective function for each candidate solution
        solutions_eval = [(sol, problem(sol)) for sol in solutions]
        # Provide the evaluation results to the tell() method
        es.tell(solutions_eval)

    # Retrieve the best solution and the ESResult object
    xopt = es._best[0]

    return xopt, es

def fmin(problem, mean, sigma):
    es = CMA(mean=mean, sigma=sigma)
    return _loop_cma_es(problem, es)


def fmin_random(problem, mean, sigma):
    initializer = RandomInitializer
    es = CMA(mean=mean, sigma=sigma, vector_initializer=initializer)
    return _loop_cma_es(problem, es)


def fmin_square_root(problem, mean, sigma):
    initializer = ReverseSquareRootInitializer

    es = CMA(mean=mean, sigma=sigma, vector_initializer=initializer)
    return _loop_cma_es(problem, es)