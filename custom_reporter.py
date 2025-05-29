import gzip
import pathlib
import pickle
import random

import neat
from neat.reporting import BaseReporter


class NewBestReport(BaseReporter):
    def __init__(self, 
                 simulation_eval,
                 checkpoint_dir:pathlib.Path,
                 fitness_target:float=1000,
                 ):
        super().__init__()
        self.best_fitness = -float("inf")
        self.fitness_target = fitness_target
        self.simulation_eval = simulation_eval
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_generation = 0

    def start_generation(self, generation):
        self.current_generation = generation
    
    def post_evaluate(self, config, population, species, best_genome):
        if self.best_fitness < best_genome.fitness:
            self.best_fitness = best_genome.fitness
            self.save_current_best(config, population, species, self.current_generation, best_genome)
            if best_genome.fitness > self.fitness_target:
                self.fitness_target = best_genome.fitness# + 200
                self.simulation_eval(best_genome, config, config.visualizer)

    def save_current_best(self, config, population:neat.Population, species_set, generation, best_genome):
        """ Save the current simulation state. """
        filename = "best_population_checkpoint"
        filepath = self.checkpoint_dir/filename

        print("Saving checkpoint to {0}".format(filename))

        with gzip.open(filepath.absolute(), 'wb', compresslevel=5) as f:
            data = (generation, config, population, species_set, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        best_genome_filename = "best_genome"
        best_genome_filepath = self.checkpoint_dir/best_genome_filename
        with gzip.open(best_genome_filepath.absolute(), 'wb', compresslevel=5) as f:
            pickle.dump(best_genome, f, protocol=pickle.HIGHEST_PROTOCOL)

        