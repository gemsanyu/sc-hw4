import torch as T
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

from problem import Problem, MinerCallback
from curriculum_full import run_full

from multiprocessing.pool import Pool
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
    
def count_total_params(model: T.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    algo = GA()
    with Pool(6) as pool:
        runner = StarmapParallelization(pool.starmap)
        problem = Problem(5541, run_full, n_replication=3, elementwise_runner=runner)
        callback = MinerCallback(run_full, "full")
        minimize(problem, algo, callback=callback, verbose=True, resample=False)
    # run_braking(x, True)
    # print(num_params)