import os
import pathlib
from functools import partial
from typing import Callable, List

import neat
import pygame
from curriculum_full import run_full
from custom_reporter import EarlyStoppingReport
from neat.parallel import ParallelEvaluator
from utils import eval_function_template
from visualizer import TrainingVisualizer


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
    population.add_reporter(EarlyStoppingReport(run_full, checkpoint_dir, fitness_target=200))
    # population.add_reporter(neat.Checkpointer(generation_interval=10))
    eval_function = partial(eval_function_template, run_full)
    evaluator = ParallelEvaluator(8, eval_function)
    # Run NEAT
    try:
        winner = population.run(evaluator.evaluate, 100)
        # print("\nTraining complete! Final best genome:")
        # print(f"Fitness: {winner.fitness:.1f}")
        # print(f"Nodes: {len(winner.nodes)}")
        # print(f"Connections: {len(winner.connections)}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    run_neat(config_file)