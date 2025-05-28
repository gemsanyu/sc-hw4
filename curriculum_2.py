import math
from typing import List, Optional, Tuple, Union

import neat
import neat.config
import pygame
from miner_objects import (BLACK, HEIGHT, WHITE, WIDTH, Asteroid, Mineral,
                           Spaceship)
from pygame.surface import Surface
from utils import apply_action, generate_inputs, generate_linear_minerals


def eval_function_2(genome: neat.DefaultGenome, 
                    config: neat.config):
    return run_simulation_curr_2(genome, config)

def run_simulation_curr_2(genome: neat.DefaultGenome, 
                          config: neat.Config, 
                          visualizer=None):
    pygame.init()
    screen = None
    if visualizer is not None:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("NEAT - Space Miner Training")
    clock = pygame.time.Clock()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    ship = Spaceship(screen)
    minerals: List[Mineral] = generate_linear_minerals(ship.x, ship.y, screen=screen)
    asteroids: List[Asteroid] = []
    alive_time = 0
    genome.fitness = 0
    idle_time = 0
    MAX_IDLE_TIME = 1000
    while True:
        alive_time += 1
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        
        
        inputs = generate_inputs(ship, minerals, asteroids)
        # Get actions from network
        output = net.activate(inputs)
        
        # Execute actions
        old_x, old_y = ship.x, ship.y
        num_minerals_mined = apply_action(ship, output, minerals)
        if old_x == ship.x and old_y == ship.y:
            idle_time += 1
        else:
            idle_time = 0
        
        genome.fitness += num_minerals_mined*10 - 0.1
        if len(minerals) < 1:  # Replenish minerals
            minerals = generate_linear_minerals(ship.x, ship.y, screen=screen)
        
        # Visualization
        if visualizer:
            screen.fill(BLACK)
            for mineral in minerals:
                mineral.draw()
            for asteroid in asteroids:
                asteroid.move()
                asteroid.draw()
            ship.draw()
            visualizer.draw_stats(screen, genome.fitness, ship.minerals, ship.fuel)
            pygame.display.flip()
            clock.tick(30)
        
        # Termination conditions
        asteroid_collision = False
        closest_asteroid = min((a for a in asteroids), 
                            key=lambda a: math.hypot(ship.x-a.x, ship.y-a.y),
                            default=None)
        if closest_asteroid is not None:        
            asteroid_collision = math.hypot(ship.x-closest_asteroid.x, ship.y-closest_asteroid.y) < ship.radius + closest_asteroid.radius
        out_of_fuel = ship.fuel <= 0
        no_minerals_left = not minerals and ship.minerals == 0
        too_idle = idle_time >= MAX_IDLE_TIME
        if too_idle:
            # genome.fitness -= 100
            break
        
        if asteroid_collision or out_of_fuel or no_minerals_left or alive_time >= 5000:
            break
    
    return genome.fitness