import math
import os
import random
import time
from typing import Callable, List, Optional, Tuple, Union

import neat
import numba as nb
import pygame
from miner_objects import WHITE


class TrainingVisualizer:
    def __init__(self, config: neat.Config, simulation_evaluation: Callable):
        self.best_fitness = -float('inf')
        self.generation = 0
        self.start_time = time.time()
        self.config = config
        self.fitness_target = 30
        self.simulation_evaluation = simulation_evaluation

    def update_generation(self, best_genome):
        self.generation += 1
        if best_genome.fitness > self.best_fitness:
            self.best_fitness = best_genome.fitness
            print(f"🔥 New best fitness: {self.best_fitness:.1f}")
            if self.best_fitness > self.fitness_target:
                self.fitness_target = self.best_fitness + 50
                self.simulation_evaluation(best_genome, self.config, visualizer=self) 
        print(f"Generation {self.generation} best: {best_genome.fitness:.1f}")

    def draw_stats(self, screen, fitness, minerals, fuel):
        stats = [
            f"Gen: {self.generation}",
            f"Fitness: {fitness:.1f}",
            f"Best: {self.best_fitness:.1f}",
            f"Minerals: {minerals}",
            f"Fuel: {fuel:.1f}"
        ]
        
        for i, stat in enumerate(stats):
            font = pygame.font.SysFont(None, 36)
            text = font.render(stat, True, WHITE)
            screen.blit(text, (10, 10 + i * 40))