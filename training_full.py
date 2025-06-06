import os
import pathlib
from functools import partial
from typing import Callable, List

import neat
import pygame
from curriculum_full import run_full
from custom_reporter import EarlyStoppingReport, NewBestReport
from parallel import ParallelEvaluator
from utils import eval_function_template
from visualizer import TrainingVisualizer


def run_neat(config_file):    
    last_checkpoint_filepath = pathlib.Path()/"checkpoints"/"maneuvering_hard"/"best_population_checkpoint"
    population = neat.Checkpointer.restore_checkpoint(last_checkpoint_filepath.absolute())
    # Reset stagnation history
    for species in population.species.species.values():
        species.last_improved = population.generation
    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    checkpoint_dir = pathlib.Path()/"checkpoints"/"full"
    log_dir = pathlib.Path()/"logs"/"full"
    population.add_reporter(EarlyStoppingReport(run_full, checkpoint_dir, fitness_target=10000))
    # population.add_reporter(neat.Checkpointer(generation_interval=10))
    eval_function = partial(eval_function_template, run_full)
    evaluator = ParallelEvaluator(12, eval_function)
    # Run NEAT
    try:
        winner = population.run(evaluator.evaluate, 10000)
    finally:
        pygame.quit()

if __name__ == "__main__":
    
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    run_neat(config_file)