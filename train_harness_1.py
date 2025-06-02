import pickle
import pygame
import neat
import math
import os
import gzip
from typing import List, Tuple, Union, Optional
from pygame.surface import Surface
import pathlib
from test_harness_1 import run_harness_1
from custom_reporter import EarlyStoppingReport
from neat.parallel import ParallelEvaluator

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    
    
    last_checkpoint_filepath = pathlib.Path()/"checkpoints"/"full"/"best_population_checkpoint"
    population = neat.Checkpointer.restore_checkpoint(last_checkpoint_filepath.absolute())
    # Reset stagnation history
    for species in population.species.species.values():
        species.last_improved = population.generation
    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    checkpoint_dir = pathlib.Path()/"checkpoints"/"harness_1"
    log_dir = pathlib.Path()/"logs"/"full"
    population.add_reporter(EarlyStoppingReport(run_harness_1, checkpoint_dir, fitness_target=10000))
    # population.add_reporter(neat.Checkpointer(generation_interval=10))
    evaluator = ParallelEvaluator(12, run_harness_1)
    # Run NEAT
    try:
        winner = population.run(evaluator.evaluate, 10000)
    finally:
        pygame.quit()