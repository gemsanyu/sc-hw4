from typing import Callable
import pathlib
import pickle
import math

import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.core.algorithm import Algorithm


class Problem(ElementwiseProblem):
    def __init__(self, 
                 n_var:int, 
                 eval_function:Callable, 
                 n_replication:int=1,
                 elementwise=True, 
                 **kwargs):
        super().__init__(n_var=n_var,
                         n_obj=1,
                         n_constr=0,
                         xl=-0.2,
                         xu=0.2,
                         elementwise=elementwise, 
                         **kwargs)
        self.eval_function = eval_function
        self.n_replication = n_replication
        self.n_var = n_var
    
    def _evaluate(self, x, out, *args, **kwargs):
        total_fitness = 0
        
        total_fitness += self.eval_function(x, False)
        avg_fitness = total_fitness/self.n_replication
        out["F"] = -avg_fitness
        
class MinerCallback(Callback):
    def __init__(self, title, evaluation_function=None):
        super().__init__()
        self.evaluation_function = evaluation_function
        self.best_fitness = -99999
        self.checkpoint_dir = pathlib.Path()/"checkpoints"/title
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_params_checkpoint_path = self.checkpoint_dir/"best_params.npy"
        self.algo_checkpoint_path = self.checkpoint_dir/"algorithm.pkl"

    def notify(self, algorithm:Algorithm):
        best_fitness = -algorithm.opt[0].F[0]
        if self.best_fitness < best_fitness:
            self.best_fitness = best_fitness 
            best = algorithm.opt[0].X
            if self.evaluation_function is not None:
                self.evaluation_function(best, True)
            print(f"Saving best params, fitness={algorithm.opt[0].F}")
            np.save(self.best_params_checkpoint_path.absolute(), best)
            with open(self.algo_checkpoint_path.absolute(), "wb") as f:
                pickle.dump(algorithm, f)