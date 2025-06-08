import pathlib

import torch as T
import numpy as np

from curriculum_braking import run_braking

    
def count_total_params(model: T.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    title="braking"
    best_params_path = pathlib.Path()/"checkpoints"/title/"best_params.npy"
    best = np.load(best_params_path.absolute())
    run_braking(best, True)
    # print(num_params)