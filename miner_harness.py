import pygame
import random
import math

from miner_objects import WIDTH, HEIGHT, WHITE, BLACK, BLUE, RED

# Player (Spaceship)
class Spaceship:
    def __init__(self, screen):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = 5
        self.angle = 0
        self.fuel = 100
        self.minerals = 0
        self.radius = 15
        self.screen = screen

    def move(self, dx, dy):
        if self.fuel > 0:
            self.x = (self.x + dx) % WIDTH
            self.y = (self.y + dy) % HEIGHT
            self.fuel -= 0.1

    def mine(self, minerals):
        num_minerals_mined = 0
        for mineral in minerals[:]:
            dist = math.hypot(self.x - mineral.x, self.y - mineral.y)
            if dist < self.radius + mineral.radius:
                minerals.remove(mineral)
                self.minerals += 1
                num_minerals_mined += 1
                self.fuel = min(100, self.fuel + 10)  # Refuel when mining
        return num_minerals_mined

    def draw(self):
        if self.screen is None:
            return
        pygame.draw.circle(self.screen, BLUE, (int(self.x), int(self.y)), self.radius)
        # Draw a triangle for direction
        points = [
            (self.x + self.radius * math.cos(self.angle), 
             self.y + self.radius * math.sin(self.angle)),
            (self.x + self.radius * math.cos(self.angle + 2.5), 
             self.y + self.radius * math.sin(self.angle + 2.5)),
            (self.x + self.radius * math.cos(self.angle - 2.5), 
             self.y + self.radius * math.sin(self.angle - 2.5))
        ]
        pygame.draw.polygon(self.screen, WHITE, points)

# Mineral
class Mineral:
    _coordinates = [(385,373), (51,268), (309,188), (123,251), (305,554), (362,78), (760,454), (124,126), (84,500), (468,479)]
    _index = 0

    @staticmethod
    def get_next_coord():
        coord = Mineral._coordinates[Mineral._index]
        Mineral._index = (Mineral._index + 1) % len(Mineral._coordinates)  # Cycle through
        return coord

    def __init__(self, screen):
        #self.x = random.randint(20, WIDTH - 20)
        #self.y = random.randint(20, HEIGHT - 20)
        self.x, self.y = Mineral.get_next_coord()  # Get next fixed coordinate
        self.radius = 10
        self.screen = screen

    def draw(self):
        if self.screen is not None:
            pygame.draw.circle(self.screen, BLUE, (self.x, self.y), self.radius)

# Asteroids (Obstacles)
class Asteroid:
    _coordinates = [(553,323), (124,303), (556,82), (490,425), (120,218), (774,142), (240,308), (64,475), (651,227), (462,77),]
    _radius = [28, 15, 15, 22, 29, 17, 24, 22, 23, 17,]
    _speed = [(1.78425438642474,0.6732917959108602), (0.3526239655425698,-1.295737594576594), (0.9002543984134839,-0.1395601523204939), 
              (-0.5781368926686516,1.349036516126493), (1.8150210404495417,1.2741322355662592), (1.2831373928635355,0.8453257857927245), 
              (-1.8228539352311386,-1.38841117002117), (-0.9822217473337984,1.8758459317074854), (0.04980714264573827,0.5986537138756898), 
              (0.8160687662014605,-1.0618283414551777)]
    _index = 0

    @staticmethod
    def get_next_coord():
        coord = Asteroid._coordinates[Asteroid._index]
        radius = Asteroid._radius[Asteroid._index]
        speed = Asteroid._speed[Asteroid._index]
        Asteroid._index = (Asteroid._index + 1) % len(Asteroid._coordinates)  # Cycle through
        return coord, radius, speed
    
    def __init__(self, screen):
        #self.x = random.randint(0, WIDTH)
        #self.y = random.randint(0, HEIGHT)
        #self.radius = random.randint(15, 30)
        #self.speed_x = random.uniform(-2, 2)
        #self.speed_y = random.uniform(-2, 2)
        (self.x, self.y), self.radius, (self.speed_x, self.speed_y) = Asteroid.get_next_coord()  # Get next fixed coordinate
        self.screen = screen

    def move(self):
        self.x = (self.x + self.speed_x) % WIDTH
        self.y = (self.y + self.speed_y) % HEIGHT

    def draw(self):
        if self.screen is not None:
            pygame.draw.circle(self.screen, RED, (int(self.x), int(self.y)), self.radius)