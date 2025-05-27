import math
import os
import random
import time
from typing import List, Optional, Tuple, Union

import pygame
from pygame.surface import Surface

WIDTH, HEIGHT, DIAG = 800, 600, 1000
# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Game Classes (same as before)
class Mineral:
    def __init__(self, screen:Optional[Surface]):
        self.x = random.randint(20, WIDTH - 20)
        self.y = random.randint(20, HEIGHT - 20)
        self.radius = 10
        self.screen: Optional[Surface] = screen

    def draw(self):
        pygame.draw.circle(self.screen, YELLOW, (self.x, self.y), self.radius)

class Spaceship:
    def __init__(self, screen:Surface):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = 5
        self.angle = 0
        self.fuel = 100.0
        self.minerals = 0
        self.radius = 15
        self.screen:Surface = screen

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
        pygame.draw.circle(self.screen, BLUE, (int(self.x), int(self.y)), self.radius)
        points = [
            (self.x + self.radius * math.cos(self.angle), 
            self.y + self.radius * math.sin(self.angle)),
            (self.x + self.radius * math.cos(self.angle + 2.5), 
            self.y + self.radius * math.sin(self.angle + 2.5)),
            (self.x + self.radius * math.cos(self.angle - 2.5), 
            self.y + self.radius * math.sin(self.angle - 2.5))
        ]
        pygame.draw.polygon(self.screen, WHITE, points)


class Asteroid:
    def __init__(self, screen:Surface):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.radius = random.randint(15, 30)
        self.speed_x = random.uniform(-2, 2)
        self.speed_y = random.uniform(-2, 2)
        self.screen:Surface = screen

    def move(self):
        self.x = (self.x + self.speed_x) % WIDTH
        self.y = (self.y + self.speed_y) % HEIGHT

    def draw(self):
        pygame.draw.circle(self.screen, RED, (int(self.x), int(self.y)), self.radius)