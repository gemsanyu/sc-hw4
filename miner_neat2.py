import math
import os
import random
import time
from typing import List

import pygame
import neat
# POOR FITNESS FUNCTION - spin on own axis

# Initialize pygame
pygame.init()
WIDTH, HEIGHT, DIAG = 800, 600, 1000
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT - Space Miner Training")
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Game Classes (same as before)

class Mineral:
    def __init__(self):
        self.x = random.randint(20, WIDTH - 20)
        self.y = random.randint(20, HEIGHT - 20)
        self.radius = 10

    def draw(self):
        pygame.draw.circle(screen, YELLOW, (self.x, self.y), self.radius)

class Spaceship:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = 5
        self.angle = 0
        self.fuel = 100
        self.minerals = 0
        self.radius = 15

    def move(self, dx:int, dy:int):
        if self.fuel > 0:
            self.x = (self.x + dx) % WIDTH
            self.y = (self.y + dy) % HEIGHT
            self.fuel -= 0.1

    def mine(self, minerals:List[Mineral])->int:
        num_minerals_mined = 0
        for mineral in minerals[:]:
            dist = math.hypot(self.x - mineral.x, self.y - mineral.y)
            if dist < self.radius + mineral.radius:
                minerals.remove(mineral)
                self.minerals += 1
                num_minerals_mined += 1
                self.fuel = min(100, self.fuel + 10)
        return num_minerals_mined
    
    
    def draw(self):
        pygame.draw.circle(screen, BLUE, (int(self.x), int(self.y)), self.radius)
        points = [
            (self.x + self.radius * math.cos(self.angle), 
            self.y + self.radius * math.sin(self.angle)),
            (self.x + self.radius * math.cos(self.angle + 2.5), 
            self.y + self.radius * math.sin(self.angle + 2.5)),
            (self.x + self.radius * math.cos(self.angle - 2.5), 
            self.y + self.radius * math.sin(self.angle - 2.5))
        ]
        pygame.draw.polygon(screen, WHITE, points)


class Asteroid:
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.radius = random.randint(15, 30)
        self.speed_x = random.uniform(-2, 2)
        self.speed_y = random.uniform(-2, 2)

    def move(self):
        self.x = (self.x + self.speed_x) % WIDTH
        self.y = (self.y + self.speed_y) % HEIGHT

    def draw(self):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), self.radius)

def apply_action(ship:Spaceship, output, minerals)->int:
    ship.angle += (output[0] * 2 - 1) * 0.1  # Turn (-1 to 1)
    if output[1] > 0.5:  # Thrust
        dx = ship.speed * math.cos(ship.angle)
        dy = ship.speed * math.sin(ship.angle)
        ship.move(dx, dy)
    num_minerals_mined: int = 0
    if output[2] > 0.5:  # Mine
        num_minerals_mined = ship.mine(minerals)
    return num_minerals_mined

def run_simulation_curr_1(genome: neat.DefaultGenome, config: neat.Config, visualizer=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    ship = Spaceship()
    minerals = [Mineral() for _ in range(1)]
    asteroids = []
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
        
        # Find closest objects
        closest_mineral = min((m for m in minerals), 
                            key=lambda m: math.hypot(ship.x-m.x, ship.y-m.y), 
                            default=None)
        closest_asteroid = min((a for a in asteroids), 
                              key=lambda a: math.hypot(ship.x-a.x, ship.y-a.y),
                              default=None)
        
        # Get inputs (handle case where all minerals are collected)
        distance_to_closest_mineral = 1
        if closest_mineral is not None:
            distance_to_closest_mineral = math.sqrt((ship.x-closest_mineral.x)**2 + (ship.y-closest_mineral.y)**2)
            distance_to_closest_mineral /= DIAG
        distance_to_closest_asteroid = 1
        if closest_asteroid is not None:
            distance_to_closest_asteroid = math.sqrt((ship.x-closest_asteroid.x)**2 + (ship.y-closest_asteroid.y)**2)
            distance_to_closest_asteroid /= DIAG
        inputs = [
            distance_to_closest_mineral,
            math.atan2(closest_mineral.y-ship.y, closest_mineral.x-ship.x)/math.pi if closest_mineral else 0,
            distance_to_closest_asteroid,
            ship.fuel / 100.0
        ]
        
        # Get actions from network
        output = net.activate(inputs)
        # print(output)
        
        # Execute actions
        old_x, old_y = ship.x, ship.y
        num_minerals_mined = apply_action(ship, output, minerals)
        if old_x == ship.x and old_y == ship.y:
            idle_time += 1
        else:
            idle_time = 0
        if len(minerals) < 1:  # Replenish minerals
            minerals.append(Mineral())
        new_distance_to_closest_mineral = 1
        if closest_mineral is not None:
            new_distance_to_closest_mineral = math.sqrt((ship.x-closest_mineral.x)**2 + (ship.y-closest_mineral.y)**2)
            new_distance_to_closest_mineral /= DIAG
        
        delta_distance = distance_to_closest_mineral - new_distance_to_closest_mineral
        genome.fitness += 5*delta_distance -0.01
        
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
        if closest_asteroid is not None:        
            asteroid_collision = math.hypot(ship.x-closest_asteroid.x, ship.y-closest_asteroid.y) < ship.radius + closest_asteroid.radius
        out_of_fuel = ship.fuel <= 0
        no_minerals_left = not minerals and ship.minerals == 0
        too_idle = idle_time >= MAX_IDLE_TIME
        if too_idle:
            # genome.fitness -= 100
            break
        
        
        if asteroid_collision or out_of_fuel or no_minerals_left: #or alive_time >= 5000:
            break
    # Calculate fitness - reward both survival and mining

def run_simulation(genome, config, visualizer=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    ship = Spaceship()
    minerals = [Mineral() for _ in range(5)]
    asteroids = [Asteroid() for _ in range(8)]
    alive_time = 0
    
    while True:
        alive_time += 1
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Find closest objects
        closest_mineral = min((m for m in minerals), 
                            key=lambda m: math.hypot(ship.x-m.x, ship.y-m.y), 
                            default=None)
        closest_asteroid = min((a for a in asteroids), 
                              key=lambda a: math.hypot(ship.x-a.x, ship.y-a.y))
        
        # Get inputs (handle case where all minerals are collected)
        inputs = [
            math.hypot(ship.x - closest_mineral.x)/WIDTH if closest_mineral else 0,
            math.atan2(closest_mineral.y-ship.y, closest_mineral.x-ship.x)/math.pi if closest_mineral else 0,
            math.hypot(ship.x - closest_asteroid.x)/WIDTH,
            ship.fuel / 100.0
        ]
        
        # Get actions from network
        output = net.activate(inputs)
        
        # Execute actions
        apply_action(ship, output, minerals)
        if len(minerals) < 3:  # Replenish minerals
            minerals.extend(Mineral() for _ in range(2))
        
        # Calculate fitness - reward both survival and mining
        genome.fitness = ship.minerals * 10 + alive_time * 0.01  # Reduced time bonus
        
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
        asteroid_collision = math.hypot(ship.x-closest_asteroid.x, ship.y-closest_asteroid.y) < ship.radius + closest_asteroid.radius
        out_of_fuel = ship.fuel <= 0
        no_minerals_left = not minerals and ship.minerals == 0
        
        if asteroid_collision or out_of_fuel or no_minerals_left or alive_time >= 5000:
            break

class TrainingVisualizer:
    def __init__(self):
        self.best_fitness = -float('inf')
        self.generation = 0
        self.start_time = time.time()
        self.font = pygame.font.SysFont(None, 36)
        
    def update_generation(self, best_genome):
        self.generation += 1
        if best_genome.fitness > self.best_fitness:
            self.best_fitness = best_genome.fitness
            print(f"🔥 New best fitness: {self.best_fitness:.1f}")
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
            text = self.font.render(stat, True, WHITE)
            screen.blit(text, (10, 10 + i * 40))

def eval_genomes(genomes, config):
    visualizer = config.visualizer
    
    # First evaluate all genomes to find the best
    best_in_generation = None
    best_fitness = -float('inf')
    
    for genome_id, genome in genomes:
        run_simulation_curr_1(genome, config, visualizer=None)  # No visualization during evaluation
        # print(genome_id,genome.fitness)
        # Track the best in this generation
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_in_generation = genome
    print(f"Current best fitness: {best_fitness}")
    # Update visualizer with this generation's results
    visualizer.update_generation(best_in_generation)
    
    # Visualize the best genome from this generation
    if best_fitness > 30:
        print(f"Displaying generation {visualizer.generation} best (Fitness: {best_fitness:.1f})")
        run_simulation_curr_1(best_in_generation, config, visualizer=visualizer)  # With visualization


def run_neat(config_file):
    # Initialize pygame
    pygame.init()
    global screen, clock, WIDTH, HEIGHT
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT - Space Miner Training")
    clock = pygame.time.Clock()
    
    # Create and store visualizer in config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    config.visualizer = TrainingVisualizer()
    
    # Create population
    population = neat.Population(config)
    
    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Run NEAT
    try:
        winner = population.run(eval_genomes, 1000)
        print("\nTraining complete! Final best genome:")
        print(f"Fitness: {winner.fitness:.1f}")
        print(f"Nodes: {len(winner.nodes)}")
        print(f"Connections: {len(winner.connections)}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    run_neat(config_file)