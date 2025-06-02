import os
import pathlib
from functools import partial
from typing import Callable, List

import neat
import pygame
from curriculum_maneuvering import run_maneuvering_1
from custom_reporter import EarlyStoppingReport
from neat.parallel import ParallelEvaluator
from utils import eval_function_template
from visualizer import TrainingVisualizer


def run_neat(config_file):    
    last_checkpoint_filepath = pathlib.Path()/"checkpoints"/"braking"/"best_population_checkpoint"
    population = neat.Checkpointer.restore_checkpoint(last_checkpoint_filepath.absolute())
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    config.visualizer = TrainingVisualizer(config, run_maneuvering_1)
    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    checkpoint_dir = pathlib.Path()/"checkpoints"/"maneuvering_easy"
    population.add_reporter(EarlyStoppingReport(run_maneuvering_1, checkpoint_dir, fitness_target=500))
    # population.add_reporter(neat.Checkpointer(generation_interval=10))
    eval_function = partial(eval_function_template, run_maneuvering_1)
    evaluator = ParallelEvaluator(12, eval_function)
    # Run NEAT
    try:
        winner = population.run(evaluator.evaluate, 1000)
    finally:
        pygame.quit()

if __name__ == "__main__":
    
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    run_neat(config_file)