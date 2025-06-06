import math
import random
from typing import List, Optional, Tuple, Union

import neat
import neat.config
import pygame
from miner_objects import (BLACK, HEIGHT, WHITE, WIDTH, Asteroid, Mineral,
                           Spaceship)
from pygame.surface import Surface
from utils import apply_action, generate_inputs


def compute_max_distance_from_ship(sx, sy, dx, dy):
    # sx = WIDTH / 2
    # sy = HEIGHT / 2

    if dx == 0:
        max_dist_x = float('inf')
    elif dx > 0:
        max_dist_x = (WIDTH - sx) / dx
    else:
        max_dist_x = -sx / dx

    if dy == 0:
        max_dist_y = float('inf')
    elif dy > 0:
        max_dist_y = (HEIGHT - sy) / dy
    else:
        max_dist_y = -sy / dy

    return min(max_dist_x, max_dist_y)

def generate_alternating_minerals_and_asteroids(sx, sy, screen)->Tuple[List[Mineral],List[Asteroid]]:
    minerals: List[Mineral] = []
    asteroids: List[Asteroid] = []
    
    # Random direction angle from the ship
    angle = random.uniform(0, 2 * math.pi)
    dx = math.cos(angle)
    dy = math.sin(angle)
    max_dist = compute_max_distance_from_ship(sx, sy, dx, dy)
    spacing = max_dist/5.5
    jitter = 10
    current_x, current_y = sx, sy
    for i in range(5):  # M-A-M-A-M = 5 steps
        # Add some jitter to make it less regular
        jitter_x = random.uniform(-jitter, jitter)
        jitter_y = random.uniform(-jitter, jitter)

        current_x += dx * spacing + jitter_x
        current_y += dy * spacing + jitter_y
    
        if i % 2 == 0:
            obj = Mineral(screen)
            obj.x = current_x
            obj.y = current_y
            minerals.append(obj)
        else:
            obj = Asteroid(screen)
            obj.x = current_x
            obj.y = current_y
            obj.speed_x = 0
            obj.speed_y = 0
            asteroids.append(obj)
    
    return minerals, asteroids
    

def run_maneuvering_1(genome: neat.DefaultGenome, 
                          config: neat.Config, 
                          visualizer: Optional[Surface]=None):
    
    screen = None
    clock = None
    if visualizer is not None:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("NEAT - Space Miner Training")
        clock = pygame.time.Clock()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    ship = Spaceship(screen)
    
    # Random direction angle from the ship
    angle = random.uniform(0, 2 * math.pi)
    dx = math.cos(angle)
    dy = math.sin(angle)
    max_dist = compute_max_distance_from_ship(ship.x, ship.y, dx, dy)
    minerals: List[Mineral] = [Mineral(screen)]
    minerals[0].x = ship.x + 0.8*max_dist*dx
    minerals[0].y = ship.y + 0.8*max_dist*dy
    asteroids: List[Asteroid] = [Asteroid(screen=screen) for _ in range(1)]
    for asteroid in asteroids:
        asteroid.speed_x = 0
        asteroid.speed_y = 0
    asteroids[0].x = ship.x + 0.5*max_dist*dx
    asteroids[0].y = ship.y + 0.5*max_dist*dy
    
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
            visualizer.draw_stats(screen, reward, ship.minerals, ship, output)
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
    elif len(minerals)==0:
        reward += 100
    if screen is not None:
        pygame.quit()
    return reward


def run_maneuvering_2(genome: neat.DefaultGenome, 
                          config: neat.Config, 
                          visualizer: Optional[Surface]=None):
   
    screen = None
    clock = None
    if visualizer is not None:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("NEAT - Space Miner Training")
        clock = pygame.time.Clock()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    ship = Spaceship(screen)
    minerals, asteroids = generate_alternating_minerals_and_asteroids(ship.x, ship.y, screen)
    minerals.extend(Mineral(screen) for _ in range(2))
    
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
            visualizer.draw_stats(screen, reward, ship.minerals, ship, output)
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
    elif len(minerals)==1:
        reward += 100
    elif len(minerals)==0:
        reward += 500
    if screen is not None:
        pygame.quit()
    return reward