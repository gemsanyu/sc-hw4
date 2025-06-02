import math
from typing import List, Optional, Tuple, Union

import neat
import numba as nb
import numpy as np
import pygame
from miner_objects import (DIAG, HEIGHT, RED, WHITE, WIDTH, YELLOW, Asteroid,
                           Mineral, Spaceship)
from pygame.surface import Surface


def eval_function_template(simulation_evaluation, genome: neat.DefaultGenome, config: neat.config, num_samples:int=3):
    total_fitness = 0
    for n in range(num_samples):
        total_fitness += simulation_evaluation(genome, config) 
    avg_fitness = total_fitness/num_samples
    return avg_fitness

@nb.njit(
    nb.float64(
        nb.float64,  # ox: ray origin x
        nb.float64,  # oy: ray origin y
        nb.float64,  # dx: ray direction x (normalized)
        nb.float64,  # dy: ray direction y (normalized)
        nb.float64,  # cx: circle center x
        nb.float64,  # cy: circle center y
        nb.float64,  # r: circle radius
        nb.float64,  # W: world width
        nb.float64   # H: world height
    )
)
def ray_circle_intersect_toroidal(ox, oy, dx, dy, cx, cy, r, W, H):
    """
    Compute the intersection of a ray (origin (ox, oy), direction (dx, dy))
    with a circle of radius r located at (cx, cy) *in a toroidal world of size W×H*.
    
    Returns:
      - The smallest non-negative t along the ray so that (ox + t·dx, oy + t·dy) 
        lies on the perimeter of some copy of the circle.
      - If no forward intersection exists, returns -9999.0.
    """

    t_min = 9999.0  # Large sentinel: “no intersection yet”

    # Loop over all 3×3 = 9 possible wrapped copies of the circle
    for dx_wrap in (-1.0, 0.0, 1.0):
        for dy_wrap in (-1.0, 0.0, 1.0):
            # Wrap the circle center by (dx_wrap * W, dy_wrap * H)
            cx_wrapped = cx + dx_wrap * W
            cy_wrapped = cy + dy_wrap * H

            # Compute quadratic coefficients for intersection ray–circle
            fx = ox - cx_wrapped
            fy = oy - cy_wrapped
            a = 1.0  # because (dx,dy) is assumed normalized
            b = 2.0 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - r * r

            # Discriminant
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                continue  # no real intersection with this wrapped copy

            sqrt_disc = math.sqrt(disc)
            # Two possible solutions along the ray
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)

            # Pick the smallest non-negative t
            if t1 >= 0.0 and t1 < t_min:
                t_min = t1
            if t2 >= 0.0 and t2 < t_min:
                t_min = t2

    if t_min < 9999.0:
        return t_min
    else:
        return -9999.0  # no forward intersection found


@nb.njit(nb.float64(nb.float64,nb.float64,nb.float64,nb.float64,nb.float64,nb.float64,nb.float64))
def ray_circle_intersect(ox:float,
                         oy:float,
                         dx:float,
                         dy:float,
                         cx:float,
                         cy:float,
                         r:float)->float:
    # 1. Compute quadratic coefficients
    fx = ox - cx
    fy = oy - cy
    a = 1  # dx*dx + dy*dy, assuming (dx,dy) normalized
    b = 2 * (fx * dx + fy * dy)
    c = fx*fx + fy*fy - r*r

    # 2. Discriminant
    disc = b*b - 4*a*c
    if disc < 0:
        return -9999.0   # no intersection

    # 3. Two possible solutions
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)

    # 4. Choose nearest forward hit
    t = -9999.0
    for t_candidate in (t1, t2):
        if t_candidate >= 0:
            if t < 999 or t_candidate < t:
                t = t_candidate

    return t  # may remain None if both < 0

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


def apply_action(ship:Spaceship, output, minerals)->int:
    # ship.angle += (output[0] * 2 - 1) * 0.1  # Turn (-1 to 1)
    d_angle = max(min(output[0], 0.5), -0.5)
    ship.angle += d_angle
    # dx, dy = 0,0
    # if output[1] > 0.5:  # Thrust
    dx = ship.speed * math.cos(ship.angle)
    dy = ship.speed * math.sin(ship.angle)
    ship.move(dx, dy)
    # num_minerals_mined: int = 0
    # if output[2] > 0.5:  # Mine
    num_minerals_mined = ship.mine(minerals)
    return num_minerals_mined

def generate_static_steroids(num_asteroids:int=5, mode:str="horizontal", screen:Optional[Surface]=None)->List[Asteroid]:
    asteroids: List[Asteroid] = [Asteroid(screen) for _ in range(5)]
    for ai, asteroid in enumerate(asteroids):
        asteroid.speed_x = 0
        asteroid.speed_y = 0
        asteroid.radius = 20
        if mode == "horizontal":
            asteroid.x = WIDTH/(num_asteroids+1) * (ai+1.5)
            asteroid.y = HEIGHT/2
        elif mode == "vertical":
            asteroid.x = WIDTH/2
            asteroid.y = HEIGHT/(num_asteroids+1) * (ai+1.5)
    return asteroids

def generate_linear_minerals(ship_x:int, ship_y:int, num_minerals:int=5, screen:Optional[Surface]=None)->List[Mineral]:
    seed_mineral = Mineral(screen)
    # Vector from ship to seed mineral
    dx = seed_mineral.x - ship_x
    dy = seed_mineral.y - ship_y

    minerals:List[Mineral] = []
    for i in range(num_minerals):
        t = i / (num_minerals - 1)  # Evenly spaced [0, 1]
        x = ship_x + t * dx
        y = ship_y + t * dy
        m = Mineral(screen)
        m.x = x
        m.y = y
        minerals.append(m)
    minerals.append(seed_mineral)
    return minerals


def cast_rays_nb(ship_x: float, 
                 ship_y: float, 
                 ship_angle: float,
                 num_rays:int, 
                 object_coords:np.ndarray, 
                 obj_radius: np.ndarray, 
                 obj_flags: np.ndarray,
                 max_length:float,
                 dist_flag_arr:np.ndarray)->np.ndarray:
    angles = ship_angle + np.arange(num_rays) * (math.pi/(num_rays/2))
    dxs, dys = np.cos(angles), np.sin(angles)
    
    num_objs = obj_radius.shape[0]

    for ri in range(num_rays):
        closest_t = -9999
        closest_flag = 0
        for i in range(num_objs):  # mix of minerals & asteroids
            t = ray_circle_intersect(ship_x, ship_y, dxs[ri], dys[ri], object_coords[i,0], object_coords[i,1], obj_radius[i])
            if t < -999:
                continue
            if closest_t is None or t < closest_t:
                closest_t = t
                closest_flag = obj_flags[i]

        if closest_t < -999:
            closest_t = 99999.
            closest_flag = 0
            # no hit
        # normalize
        norm_dist = min(1.0, closest_t / max_length)
        dist_flag_arr[ri*2] = norm_dist
        dist_flag_arr[ri*2+1] = closest_flag
        
    return dist_flag_arr

def cast_ray_nb_caller(ship_x:float, ship_y:float, ship_angle:float, num_rays:int, objects:List[Union[Mineral, Asteroid]])->np.ndarray:
    dist_flag_arr = np.empty((num_rays*2,), dtype=float)
    object_coords = np.asanyarray([[obj.x, obj.y] for obj in objects], dtype=float)
    obj_radius = np.asanyarray([obj.radius for obj in objects], dtype=float)
    obj_flags = []
    for obj in objects:
        if isinstance(obj, Mineral):
            obj_flags.append(1.)
        else:
            obj_flags.append(-1.)
    obj_flags = np.asanyarray(obj_flags)
    dist_flag_arr = cast_rays_nb(ship_x, ship_y, ship_angle, num_rays, object_coords, obj_radius, obj_flags, DIAG, dist_flag_arr)    
    return dist_flag_arr.tolist()

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
    
    
    # closest_mineral = min((m for m in minerals), 
    #                     key=lambda m: math.hypot(ship.x-m.x, ship.y-m.y), 
    #                     default=None)
    # if closest_mineral is not None:
    #     dx = closest_mineral.x-ship.x
    #     dy = closest_mineral.y-ship.y
    #     m_angle = math.atan2(dx,dy)
    #     dist = math.hypot(closest_mineral.x-ship.x, closest_mineral.y-ship.y)/DIAG
    #     inputs += [dist, math.sin(m_angle), math.cos(m_angle)]
    # else:
    #     inputs += [1, 0, 0]
    
    inputs.append(ship.fuel/100.0)
    return inputs
