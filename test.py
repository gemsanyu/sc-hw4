import copy
import gzip
import os
import pathlib
import pickle
from functools import partial
from typing import Callable, List

import neat
import pygame
from curriculum_1 import run_simulation_curr_1
from custom_reporter import NewBestReport
from neat.config import Config
from neat.genome import DefaultGenome
from neat.parallel import ParallelEvaluator
from utils import eval_function_template
from visualizer import TrainingVisualizer


def run_neat(config_file):    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    config.visualizer = TrainingVisualizer(config, run_simulation_curr_1)
    
    checkpoint_dir = pathlib.Path()/"checkpoints"/"curriculum_2"
    population = neat.Population(config)
    
    
    best_genome_filepath = pathlib.Path()/"checkpoints"/"curriculum_1"/"best_genome"
    best_genome = None
    with gzip.open(best_genome_filepath.absolute()) as f:
        best_genome = pickle.load(f)
    run_simulation_curr_1(best_genome, config, config.visualizer)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    run_neat(config_file)