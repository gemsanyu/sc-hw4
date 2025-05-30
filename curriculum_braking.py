import math
from typing import List, Optional, Tuple, Union

import neat
import neat.config
import pygame
from miner_objects import (BLACK, HEIGHT, WHITE, WIDTH, Asteroid, Mineral,
                           Spaceship)
from pygame.surface import Surface
from utils import apply_action, generate_inputs, generate_linear_minerals


def generate_minerals_in_front_of_asteroid(sx, sy, ax, ay, screen, num_minerals:int)->List[Mineral]:
    minerals: List[Mineral] = [Mineral(screen) for _ in range(num_minerals)]
    dx, dy = ax - sx, ay-sy
    num_minerals = len(minerals)
    for i, mineral in enumerate(minerals):
        mineral.x = sx + dx*0.8*(i+1)/num_minerals
        mineral.y = sy + dy*0.8*(i+1)/num_minerals
    return minerals
    

def run_braking(genome: neat.DefaultGenome, 
                          config: neat.Config, 
                          visualizer: Optional[Surface]=None):
    pygame.init()
    screen = None
    if visualizer is not None:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("NEAT - Space Miner Training")
    clock = pygame.time.Clock()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    ship = Spaceship(screen)
    asteroids: List[Asteroid] = [Asteroid(screen=screen) for _ in range(1)]
    while math.hypot(ship.x-asteroids[0].x, ship.y-asteroids[0].y) <= 10*ship.radius:
        asteroids: List[Asteroid] = [Asteroid(screen=screen) for _ in range(1)]
    for asteroid in asteroids:
        asteroid.speed_x = 0
        asteroid.speed_y = 0
    minerals: List[Mineral] = generate_minerals_in_front_of_asteroid(ship.x, ship.y, asteroids[0].x, asteroids[0].y, screen, num_minerals=3)
    
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
        inputs = generate_inputs(ship, minerals, asteroids, screen)
        # Get actions from network
        output = net.activate(inputs)
        
        # Execute actions
        old_x, old_y = ship.x, ship.y
        num_minerals_mined = apply_action(ship, output, minerals)
        if old_x == ship.x and old_y == ship.y:
            idle_time += 1
        else:
            idle_time = 0
        
        reward += num_minerals_mined*20

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
            reward = -500
            break
        if asteroid_collision:
            reward -= 500
            break
        # or no_minerals_left
        if len(minerals)>0:
            reward -= 0.1
        if out_of_fuel  or alive_time >= 1000:
            break
    if ship.minerals < 1:
        reward -= 500
    elif ship.minerals == 3:
        reward += 100
    pygame.quit()
    return reward