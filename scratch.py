from utils import generate_static_steroids, WIDTH

if __name__ == "__main__":
    asteroids = generate_static_steroids(5)
    print(WIDTH)
    for asteroid in asteroids:
        print(asteroid.x)