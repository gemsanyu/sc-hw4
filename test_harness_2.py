import gzip
import math
import os
import pickle
from typing import List, Optional, Tuple, Union

import neat
import pygame
from miner_objects import BLACK, BLUE, DIAG, HEIGHT, RED, WHITE, WIDTH, YELLOW
from pygame.surface import Surface
from utils import apply_action, ray_circle_intersect_toroidal

#from miner import Spaceship, Mineral, Asteroid  # Import your game classes random
from miner_harness2 import (  # Import your game classes fixed locations
    Asteroid, Mineral, Spaceship)

# from miner_harness2 import Spaceship, Mineral, Asteroid 
#from miner_harness3 import Spaceship, Mineral, Asteroid 




def load_trained_model(filename):
    """
    Load a saved genome and config
    """
    with gzip.open(filename) as f:
        best_genome = pickle.load(f)
    
    print(f"Model loaded from {filename}")
    return best_genome


def cast_ray(ox:float,
             oy:float,
             angle:float,
             objects:List[Union[Mineral, Asteroid]])->Tuple[float, int]:
    dx, dy = math.cos(angle), math.sin(angle)
    closest_t = None
    closest_flag = 0

    for obj in objects:  # mix of minerals & asteroids
        t = ray_circle_intersect_toroidal(ox, oy, dx, dy, obj.x, obj.y, obj.radius, WIDTH, HEIGHT)
        if t < -999:
            continue
        if closest_t is None or t < closest_t:
            closest_t = t
            closest_flag = +1 if isinstance(obj, Mineral) else -1

    if closest_t is None:
        return 1.0, 0     # no hit
    # normalize
    norm_dist = min(1.0, closest_t / DIAG)
    return norm_dist, closest_flag

def generate_inputs(ship: Spaceship, minerals: List[Mineral], asteroids: List[Asteroid], screen: Optional[Surface]=None)->List[Union[int, float]]:
    inputs = []
    num_rays = 24
    for i in range(num_rays):
        ray_angle = ship.angle + i * (math.pi/(num_rays/2))
        dist, flag = cast_ray(ship.x, ship.y, ray_angle, asteroids + minerals)
        dx = math.cos(ray_angle) * dist * DIAG
        dy = math.sin(ray_angle) * dist * DIAG
        end_x = ship.x + dx
        end_y = ship.y + dy
        if screen is not None:
            color = WHITE
            if flag <0:
                color = RED
            elif flag>0:
                color = YELLOW
            pygame.draw.line(screen, color, (ship.x, ship.y), (end_x, end_y), 1)
        inputs.append(dist)
        inputs.append(flag)
    inputs += [math.sin(ship.angle), math.cos(ship.angle)]
    
    inputs.append(ship.fuel/100.0)
    return inputs

def run_harness_2(genome: neat.DefaultGenome, config: neat.Config, visualize=False):
    # Initialize pygame
    Mineral._index = 0
    Asteroid._index = 0
    screen = None
    clock = None
    if visualize:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Trained Miner Ship")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 24)
    
    successful_minings=0
    # Create the neural network
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Game setup
    ship = Spaceship(screen)
    minerals = [Mineral(screen) for _ in range(5)]
    asteroids = [Asteroid(screen) for _ in range(len(Asteroid._coordinates))]
    
    alive_time=0

    running = True
    while running:
        alive_time+=1
        if screen is not None:
            screen.fill(BLACK)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        
        
        inputs = generate_inputs(ship, minerals, asteroids)
        
        # Get network output
        output = net.activate(inputs)
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

        # Draw everything
        

        if screen is not None:
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

if __name__ == "__main__":
    # Load the trained model
    genome = load_trained_model('checkpoints/harness_2/best_genome')
    # Create and store visualizer in config
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    run_harness_2(genome, config, True)