import math
from typing import List, Optional, Tuple, Union
import random

import torch
import neat
import neat.config
import pygame
from miner_objects import (BLACK, HEIGHT, WHITE, WIDTH, Asteroid, Mineral,
                           Spaceship)
from pygame.surface import Surface
from utils import apply_action, generate_inputs, setup_encoder

@torch.no_grad()
def run_full(genome: neat.DefaultGenome, 
                config: neat.Config, 
                visualizer: Optional[Surface]=None):
    
    torch.set_num_threads(1)       # Limit intra-op parallelism (e.g., matrix mult)
    # torch.set_num_interop_threads(1) 
    pygame.init()
    screen = None
    if visualizer is not None:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("NEAT - Space Miner Training")
    clock = pygame.time.Clock()
    encoder = setup_encoder()
    encoder.eval()
    policy = neat.nn.FeedForwardNetwork.create(genome, config)
    ship = Spaceship(screen)
    asteroids = [Asteroid(screen) for _ in range(8)]
    minerals = [Mineral(screen) for _ in range(5)]
    
    alive_time = 0
    idle_time = 0
    MAX_IDLE_TIME = 300
    reward = 0.
    while True:
        alive_time += 1
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        
        if screen is not None:
            screen.fill(BLACK)
        ship_data, asteroids_data, minerals_data = generate_inputs(ship, minerals, asteroids)
        obj_embeds, graph_embeds = encoder(ship_data, asteroids_data, minerals_data)
        ship_embed = obj_embeds[0]
        graph_embed = graph_embeds[0]
        inputs = torch.cat((ship_embed, graph_embed)).tolist()
        # Get actions from network
        output = policy.activate(inputs)
        
        # Execute actions
        old_x, old_y = ship.x, ship.y
        num_minerals_mined = apply_action(ship, output, minerals)
        if old_x == ship.x and old_y == ship.y:
            reward -= 0.2
            idle_time += 1
        else:
            idle_time = 0
        
        reward += num_minerals_mined*20
        if len(minerals) <= 2:
            while len(minerals) < 5:
                minerals.append(Mineral(screen))

        # Visualization
        if screen is not None:
            # screen.fill(BLACK)
            for mineral in minerals:
                mineral.draw()
            for asteroid in asteroids:
                asteroid.move()
                asteroid.draw()
            ship.draw()
            visualizer.draw_stats(screen, reward, ship.minerals, ship)
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
        # no_minerals_left = not minerals and ship.minerals == 0
        too_idle = idle_time >= MAX_IDLE_TIME
        if too_idle:
            reward += -500
            break
        if asteroid_collision:
            reward -= 500
            break
        if out_of_fuel or alive_time >= 5000:
            break
    pygame.quit()
    return reward