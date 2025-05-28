import os
from typing import Callable, List

import neat
import pygame
from curriculum_1 import eval_function_1, run_simulation_curr_1
from custom_reporter import NewBestReport
from neat.parallel import ParallelEvaluator
from visualizer import TrainingVisualizer

def eval_genomes(genomes:List[neat.DefaultGenome], config:neat.Config, simulation_evaluation: Callable):
    # First evaluate all genomes to find the best
    best_in_generation = None
    best_fitness = -float('inf')
    
    for genome_id, genome in genomes:
        simulation_evaluation(genome, config, visualizer=None)  # No visualization during evaluation
        # print(genome_id,genome.fitness)
        # Track the best in this generation
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_in_generation = genome
    print(f"Current best fitness: {best_fitness}")
    # Update visualizer with this generation's results
    # config.visualizer.update_generation(best_in_generation)

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
    population.add_reporter(NewBestReport(run_simulation_curr_1))
    population.add_reporter(neat.Checkpointer(generation_interval=10))
    
    
    evaluator = ParallelEvaluator(4, eval_function_1)
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