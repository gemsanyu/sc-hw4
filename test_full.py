import gzip
import os
import pathlib
import pickle

import neat
from curriculum_full import run_full
from visualizer import TrainingVisualizer


def run_neat(config_file):    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    config.visualizer = TrainingVisualizer(config, run_full)
    
    
    best_genome_filepath = pathlib.Path()/"checkpoints"/"maneuvering_hard"/"best_genome"
    best_genome = None
    with gzip.open(best_genome_filepath.absolute()) as f:
        best_genome = pickle.load(f)
    run_full(best_genome, config, config.visualizer)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    run_neat(config_file)