import os
import pathlib
from typing import Callable, List
from functools import partial

import neat
import pygame
from curriculum_1 import run_simulation_curr_1
from custom_reporter import NewBestReport
from neat.parallel import ParallelEvaluator
from visualizer import TrainingVisualizer
from utils import eval_function_template


def run_neat(config_file):    
    # Create and store visualizer in config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    config.visualizer = TrainingVisualizer(config, run_simulation_curr_1)
    # Create population
    population = neat.Population(config)
    
    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    checkpoint_dir = pathlib.Path()/"checkpoints"/"curriculum_1"
    population.add_reporter(NewBestReport(run_simulation_curr_1, checkpoint_dir, fitness_target=0))
    # population.add_reporter(neat.Checkpointer(generation_interval=10))
    eval_function = partial(eval_function_template, run_simulation_curr_1)
    evaluator = ParallelEvaluator(1, eval_function)
    # Run NEAT
    try:
        winner = population.run(evaluator.evaluate, 1000)
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