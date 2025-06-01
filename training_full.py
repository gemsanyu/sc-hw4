import os
import pathlib
from functools import partial
from typing import Callable, List

import neat
import pygame
from curriculum_full import run_full
from custom_reporter import NewBestReport
from neat.parallel import ParallelEvaluator
from utils import eval_function_template
from visualizer import TrainingVisualizer

from encoder import Encoder


def run_neat(config_file):    
    # Create and store visualizer in config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    config.visualizer = TrainingVisualizer(config, run_full)
    # Create population
    population = neat.Population(config)
    
    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    checkpoint_dir = pathlib.Path()/"checkpoints"/"full"
    log_dir = pathlib.Path()/"logs"/"full"
    population.add_reporter(NewBestReport(checkpoint_dir, log_dir))
    # population.add_reporter(neat.Checkpointer(generation_interval=10))
    eval_function = partial(eval_function_template, run_full)
    evaluator = ParallelEvaluator(8, eval_function)
    # Run NEAT
    try:
        winner = population.run(evaluator.evaluate, 10000)
    finally:
        pygame.quit()

if __name__ == "__main__":
    
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    run_neat(config_file)