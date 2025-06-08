﻿from multiprocessing.pool import Pool

import torch as T
from curriculum_braking import run_braking
from policy import Policy
from problem import MinerCallback, Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize


def count_total_params(model: T.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    policy = Policy(51)
    num_params = count_total_params(policy)
    algo = GA()
    with Pool(12) as pool:
        runner = StarmapParallelization(pool.starmap)
        problem = Problem(num_params, run_braking, n_replication=3, elementwise_runner=runner)
        callback = MinerCallback("braking", None)
        minimize(problem, algo, callback=callback, verbose=True, resample=False)
    # run_braking(x, True)
    # print(num_params)