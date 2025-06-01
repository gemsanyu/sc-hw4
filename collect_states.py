import math
import random
import pathlib
import os
import json


import pygame
from miner_objects import Spaceship, Mineral, Asteroid, BLACK, WIDTH, HEIGHT, WHITE

# Game Setup
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Space Miner")
    clock = pygame.time.Clock()
    ship = Spaceship(screen)
    minerals = [Mineral(screen) for _ in range(5)]
    num_asteroids = random.randint(3,8)
    asteroids = [Asteroid(screen) for _ in range(num_asteroids)]

    dataset_dir = pathlib.Path()/"datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    episode = 0
    episode_data_filepath = dataset_dir/f"episode_{episode}.json"
    while os.path.isfile(episode_data_filepath):
        episode += 1
        episode_data_filepath = dataset_dir/f"episode_{episode}.json"
    
    timestep = 0
    data_dict = []
    running = True
    score = 0
    while running:
        timestep += 1
        screen.fill(BLACK)
        timestep_data = {
            "ship": {"x": ship.x, "y": ship.y, "fuel": ship.fuel, "angle":ship.angle},
            "asteroids": [],
            "minerals": []
        }

        for asteroid in asteroids:
            asteroid_dict = {
                "x": asteroid.x, "y": asteroid.y,
                "speed_x": asteroid.speed_x, "speed_y": asteroid.speed_y,
                "radius": asteroid.radius
            }
            timestep_data["asteroids"].append(asteroid_dict)

        for mineral in minerals:
            mineral_dict = {"x": mineral.x, "y": mineral.y}
            timestep_data["minerals"].append(mineral_dict)

        data_dict.append(timestep_data)
            
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Player controls
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0
        if keys[pygame.K_LEFT]:
            ship.angle -= 0.1
        if keys[pygame.K_RIGHT]:
            ship.angle += 0.1
        if keys[pygame.K_UP]:
            dx = ship.speed * math.cos(ship.angle)
            dy = ship.speed * math.sin(ship.angle)
        ship.move(dx, dy)

        # Mining
        if keys[pygame.K_SPACE]:
            ship.mine(minerals)
            if len(minerals) < 3:  # Spawn new minerals if too few
                minerals.append(Mineral(screen))

        # Asteroid movement
        for asteroid in asteroids:
            asteroid.move()
            # Collision detection
            dist = math.hypot(ship.x - asteroid.x, ship.y - asteroid.y)
            if dist < ship.radius + asteroid.radius:
                running = False

        # Draw everything
        for mineral in minerals:
            mineral.draw()
        for asteroid in asteroids:
            asteroid.draw()
        ship.draw()

        # Display fuel and minerals
        font = pygame.font.SysFont(None, 36)
        fuel_text = font.render(f"Fuel: {ship.fuel:.1f}", True, WHITE)
        minerals_text = font.render(f"Minerals: {ship.minerals}", True, WHITE)
        angle_text = font.render(f"Angle: {ship.angle}", True, WHITE)
        screen.blit(fuel_text, (10, 10))
        screen.blit(minerals_text, (10, 50))
        screen.blit(angle_text, (10, 80))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    with open(episode_data_filepath.absolute(), "w") as f:
        json.dump(data_dict, f, indent=2)


if __name__ == "__main__":
    main()