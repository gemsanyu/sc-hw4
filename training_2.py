import copy
import gzip
import os
import pathlib
import pickle
from functools import partial
from typing import Callable, List

import neat
import pygame
from curriculum_2 import run_simulation_curr_2
from custom_reporter import NewBestReport
from neat.config import Config
from neat.genome import DefaultGenome
from neat.parallel import ParallelEvaluator
from utils import eval_function_template
from visualizer import TrainingVisualizer


def run_neat(config_file):    
    last_checkpoint_filepath = pathlib.Path()/"checkpoints"/"curriculum_1"/"best_population_checkpoint"
    old_population = neat.Checkpointer.restore_checkpoint(last_checkpoint_filepath.absolute())
    population_dict = old_population.population
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    config.visualizer = TrainingVisualizer(config, run_simulation_curr_2)
    
    checkpoint_dir = pathlib.Path()/"checkpoints"/"curriculum_2"
    population = neat.Population(config)
    
    
    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    checkpoint_dir = pathlib.Path()/"checkpoints"/"curriculum_2"
    population.add_reporter(NewBestReport(run_simulation_curr_2, checkpoint_dir, fitness_target=-200))
    
    # best_genome_filepath = pathlib.Path()/"checkpoints"/"curriculum_1"/"best_genome"
    # best_genome = None
    # with gzip.open(best_genome_filepath.absolute()) as f:
    #     best_genome = pickle.load(f)
    # new_population = {}
    # for i in range(config.pop_size-1):
    #     g = copy.deepcopy(best_genome)
    #     g.key = i
    #     g.mutate(config.genome_config)
    #     new_population[i]=g
    # best_genome.key = config.pop_size-1
    # new_population[config.pop_size-1] = best_genome
    species_set = config.species_set_type(config.species_set_config, population.reporters)
    species_set.speciate(config, population_dict, generation=0)
    population.population = population_dict
    population.species = species_set
    
    eval_function = partial(eval_function_template, run_simulation_curr_2)
    evaluator = ParallelEvaluator(6, eval_function)
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