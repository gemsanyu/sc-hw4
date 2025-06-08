import gzip
import math
import os
import pickle
from typing import List, Optional, Tuple, Union
import pathlib

import torch as T
import numpy as np
import pygame
#from miner import Spaceship, Mineral, Asteroid  # Import your game classes random
from miner_harness import (  # Import your game classes fixed locations
    Asteroid, Mineral, Spaceship)
from miner_objects import BLACK, BLUE, DIAG, HEIGHT, RED, WHITE, WIDTH, YELLOW
from pygame.surface import Surface
from utils import apply_action, ray_circle_intersect_toroidal, assign_params, generate_inputs
from policy import Policy
# from miner_harness2 import Spaceship, Mineral, Asteroid 
#from miner_harness3 import Spaceship, Mineral, Asteroid 


@T.no_grad()
def run_harness_1(x: np.ndarray, is_visualize: bool=False):
    Asteroid._index = 0
    Mineral._index = 0
    # Initialize pygame
    x = T.from_numpy(x)
    screen = None
    clock = None
    if is_visualize:    
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Trained Miner Ship")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 24)
    
    successful_minings=0
    # Create the neural network
    net = Policy(51)
    net.eval()
    assign_params(net, x)
    
    # Game setup
    ship = Spaceship(screen)
    minerals = [Mineral(screen) for _ in range(5)]
    asteroids = [Asteroid(screen) for _ in range(len(Asteroid._coordinates))]
    alive_time=0

    running = True
    while running:
        alive_time+=1
        if alive_time >= 20000:
            running = False
        
        if screen is not None:
            screen.fill(BLACK)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
        
        inputs = generate_inputs(ship, minerals, asteroids)
        
        # Get network output
        output = net(inputs)
        num_minerals_mined = apply_action(ship, output, minerals)
        successful_minings += num_minerals_mined
        if len(minerals) < 3:  # Spawn new minerals if too few
                minerals.append(Mineral(screen))
        
        if ship.fuel <=0:
            running = False
        # Asteroid movement
        for asteroid in asteroids:
            asteroid.move()
            # Collision detection
            dist = math.hypot(ship.x - asteroid.x, ship.y - asteroid.y)
            if dist < ship.radius + asteroid.radius:
                running = False


        if screen is not None:
            
            # Draw everything
            for mineral in minerals:
                mineral.draw()
            for asteroid in asteroids:
                asteroid.draw()
            ship.draw()
            # Display fuel and minerals
            font = pygame.font.SysFont(None, 36)
            
            # Display stats
            stats = [
                f"Minerals: {ship.minerals}",
                f"Alive Time: {alive_time:0.1f}",
                f"Fuel: {ship.fuel:.1f}",
                f"Score: {ship.minerals*100+alive_time/4:.1f}",
            ]
            for i, stat in enumerate(stats):
                text = font.render(stat, True, (255, 255, 255))
                screen.blit(text, (10, 10 + i * 25))
            
            pygame.display.flip()
            clock.tick(30)

    running=True
    while running and screen is not None:
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if screen is not None:
            screen.fill(BLACK)

        
            stats = [
                f"Minerals: {ship.minerals}",
                f"Alive Time: {alive_time:0.1f}",
                f"Fuel: {ship.fuel:.1f}",
                f"Score: {ship.minerals*100+alive_time/4:.1f}",
            ]
            for i, stat in enumerate(stats):
                text = font.render(stat, True, (255, 255, 255))
                screen.blit(text, (10, 10 + i * 25))
            text = font.render("END", True, (255, 4, 5))
            screen.blit(text, (int(HEIGHT/2), WIDTH/2))

            pygame.display.flip()
        clock.tick(60)
    score = ship.minerals*100+alive_time/4
    # print(f"Final Score:{score:.1f}")   
    pygame.quit()
    return score

def count_total_params(model: T.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    title="full"
    best_params_path = pathlib.Path()/"checkpoints"/title/"best_params.npy"
    best = np.load(best_params_path.absolute())
    run_harness_1(best, True)
    # print(num_params)