import os
import pathlib
from typing import Callable, List

import neat
import pygame
from curriculum_1 import eval_function_1, run_simulation_curr_1
from custom_reporter import NewBestReport
from neat.parallel import ParallelEvaluator
from visualizer import TrainingVisualizer


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
    population.add_reporter(NewBestReport(run_simulation_curr_1, checkpoint_dir))
    # population.add_reporter(neat.Checkpointer(generation_interval=10))
    evaluator = ParallelEvaluator(6, eval_function_1)
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