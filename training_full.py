﻿import pathlib
import pickle
from multiprocessing.pool import Pool

import numpy as np
import torch as T
from curriculum_full import run_full
from policy import Policy
from problem import MinerCallback, Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination


def count_total_params(model: T.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    policy = Policy(51)
    num_params = count_total_params(policy)
    algo_checkpoint_path = pathlib.Path()/"checkpoints"/"maneuvering_hard"/"algorithm.pkl"
    algo = None
    with Pool(12) as pool:
        runner = StarmapParallelization(pool.starmap)
        problem = Problem(num_params, run_full, n_replication=3, elementwise_runner=runner)
        with open(algo_checkpoint_path.absolute(), "rb") as f:
            algo = pickle.load(f)
        algo.termination = MaximumGenerationTermination(1000)
        algo.problem = problem
        algo.opt = None
        algo.pop.set("F", np.zeros((len(algo.pop), 1)))
        callback = MinerCallback("maneuvering_hard", run_full)
        algo.callback = callback
        minimize(problem, algo, callback=callback, copy_algorithm=False, verbose=True, resample=False)
    # run_braking(x, True)
    # print(num_params)