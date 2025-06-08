import math
import random
from typing import List, Optional, Tuple, Union

from policy import Policy
import pygame
from miner_objects import (BLACK, HEIGHT, WHITE, WIDTH, Asteroid, Mineral,
                           Spaceship)
from pygame.surface import Surface
from utils import apply_action, generate_inputs, assign_params

import torch as T
import numpy as np

@T.no_grad()
def run_full(x:np.ndarray, is_visualize:bool=False):
    x = T.from_numpy(x)
    screen = None
    clock = None
    if is_visualize:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("NEAT - Space Miner Training")
        clock = pygame.time.Clock()
    net = Policy(51)
    net.eval()
    assign_params(net, x)
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
        if screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        
        
        if screen is not None:
            screen.fill(BLACK)
        inputs = generate_inputs(ship, minerals, asteroids, screen)
        # if inputs[1] == 1:
        #     reward += max((1-inputs[0])*0.5, 0.01)
        # Get actions from network
        output = net(inputs)
        
        # Execute actions
        old_x, old_y = ship.x, ship.y
        num_minerals_mined = apply_action(ship, output, minerals)
        if old_x == ship.x and old_y == ship.y:
            # reward -= 0.2
            idle_time += 1
        else:
            idle_time = 0
        
        reward += num_minerals_mined*100
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
            stats = [
                f"Ship Position {ship.x:.1f},{ship.y:.1f}",
                f"Fitness: {reward:.1f}",
                # f"Best: {self.best_fitness:.1f}",
                f"Minerals: {minerals}",
                f"Fuel: {ship.fuel:.1f}",
                f"Output[0]: {output[0]:.2f}"
                # f"Output[1]: {output[1]:.2f}"
            ]
            
            for i, stat in enumerate(stats):
                font = pygame.font.SysFont(None, 36)
                text = font.render(stat, True, WHITE)
                screen.blit(text, (10, 10 + i * 40))
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
            # reward -= 500
            break
        # or no_minerals_left
        # if len(minerals)>0:
        #     reward -= 0.1
        if out_of_fuel or alive_time >= 5000:
            break
    # if ship.minerals < 1:
    #     reward -= 500
    # elif len(minerals)==1:
    #     reward += 100
    # elif len(minerals)==0:
    #     reward += 500
    if screen is not None:
        pygame.quit()
    reward += alive_time/4
    return reward