import os
import sys
import time
import random
import numpy as np

# ---------------- SETTINGS ----------------

MAX_GENERATIONS = 20
SECONDS_PER_GENERATION = 12
FPS = 12
MUTATION_RATE = 0.15

POPULATION_SIZE = 50
FOOD_COUNT = 25
PREDATOR_COUNT = 4

# --- ابعاد ثابت و بزرگ برای نمایش بهتر ---
WORLD_WIDTH = 100
WORLD_HEIGHT = 30

# ---------------- TERMINAL ----------------

def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

# ---------------- NEURAL NETWORK ----------------

class NeuralNetwork:
    def __init__(self, weights=None):
        if weights is None:
            self.w1 = np.random.randn(6, 8)
            self.w2 = np.random.randn(8, 2)
        else:
            self.w1 = weights[0].copy()
            self.w2 = weights[1].copy()

    def predict(self, inputs):
        hidden = np.tanh(np.dot(inputs, self.w1))
        output = np.tanh(np.dot(hidden, self.w2))
        return output

    def copy(self):
        return NeuralNetwork((self.w1, self.w2))


# ---------------- CREATURE ----------------

class Creature:
    def __init__(self, brain=None):
        self.x = random.randint(1, WORLD_WIDTH - 2)
        self.y = random.randint(1, WORLD_HEIGHT - 2)

        self.brain = brain if brain else NeuralNetwork()

        self.alive = True
        self.food_eaten = 0
        self.age = 0

    def sense_and_act(self, foods, predators, friends):
        if not self.alive:
            return

        self.age += 1

        if foods:
            food = min(
                foods,
                key=lambda f: (f[0] - self.x) ** 2 + (f[1] - self.y) ** 2
            )
            fx = (food[0] - self.x) / WORLD_WIDTH
            fy = (food[1] - self.y) / WORLD_HEIGHT
        else:
            fx = fy = 0

        if predators:
            predator = min(
                predators,
                key=lambda p: (p.x - self.x) ** 2 + (p.y - self.y) ** 2
            )
            px = (predator.x - self.x) / WORLD_WIDTH
            py = (predator.y - self.y) / WORLD_HEIGHT
        else:
            px = py = 0

        if friends:
            friend = min(
                friends,
                key=lambda c: (c.x - self.x) ** 2 + (c.y - self.y) ** 2
            )
            cx = (friend.x - self.x) / WORLD_WIDTH
            cy = (friend.y - self.y) / WORLD_HEIGHT
        else:
            cx = cy = 0

        inputs = np.array([fx, fy, px, py, cx, cy])
        output = self.brain.predict(inputs)

        self.x += int(np.sign(output[0]))
        self.y += int(np.sign(output[1]))

        self.x = max(0, min(WORLD_WIDTH - 1, self.x))
        self.y = max(0, min(WORLD_HEIGHT - 1, self.y))

        for food in foods[:]:
            if self.x == food[0] and self.y == food[1]:
                self.food_eaten += 1
                foods.remove(food)
                foods.append((
                    random.randint(0, WORLD_WIDTH - 1),
                    random.randint(0, WORLD_HEIGHT - 1)
                ))
                break

    def fitness(self):
        return self.food_eaten * 10 + self.age * 0.1


# ---------------- PREDATOR ----------------

class Predator:
    def __init__(self):
        self.x = random.randint(0, WORLD_WIDTH - 1)
        self.y = random.randint(0, WORLD_HEIGHT - 1)

    def hunt(self, creatures):
        alive = [c for c in creatures if c.alive]

        if not alive:
            return

        target = min(
            alive,
            key=lambda c: (c.x - self.x) ** 2 + (c.y - self.y) ** 2
        )

        self.x += int(np.sign(target.x - self.x))
        self.y += int(np.sign(target.y - self.y))

        self.x = max(0, min(WORLD_WIDTH - 1, self.x))
        self.y = max(0, min(WORLD_HEIGHT - 1, self.y))

        if abs(target.x - self.x) <= 1 and abs(target.y - self.y) <= 1:
            target.alive = False


# ---------------- EVOLUTION ----------------

def create_next_generation(creatures):
    ranked = sorted(
        creatures,
        key=lambda c: c.fitness(),
        reverse=True
    )
    parents = ranked[:max(2, POPULATION_SIZE // 2)]
    new_population = []

    for creature in parents[:5]:
        new_population.append(Creature(creature.brain.copy()))

    while len(new_population) < POPULATION_SIZE:
        p1, p2 = random.sample(parents, 2)

        w1 = (p1.brain.w1 + p2.brain.w1) / 2
        w2 = (p1.brain.w2 + p2.brain.w2) / 2

        w1 += np.random.randn(*w1.shape) * MUTATION_RATE
        w2 += np.random.randn(*w2.shape) * MUTATION_RATE

        child_brain = NeuralNetwork((w1, w2))
        new_population.append(Creature(child_brain))
    return new_population


# ---------------- DRAW WORLD ----------------

def draw_world(creatures, foods, predators, generation, frame):
    grid = [
        [" " for _ in range(WORLD_WIDTH)]
        for _ in range(WORLD_HEIGHT)
    ]

    for x, y in foods:
        grid[y][x] = "\033[92m.\033[0m"

    for predator in predators:
        grid[predator.y][predator.x] = "\033[91mX\033[0m"

    for creature in creatures:
        if creature.alive:
            grid[creature.y][creature.x] = "\033[96mO\033[0m"

    alive_count = sum(c.alive for c in creatures)
    total_food = sum(c.food_eaten for c in creatures)

    clear_screen()

    print("=" * WORLD_WIDTH)
    print(f"  GENERATION: {generation}/{MAX_GENERATIONS}")
    print(f"  FRAME: {frame}")
    print(f"  ALIVE: {alive_count}/{POPULATION_SIZE}")
    print(f"  FOOD EATEN: {total_food}")
    print("=" * WORLD_WIDTH)

    print("+" + "-" * WORLD_WIDTH + "+")

    for row in grid:
        print("|" + "".join(row) + "|")

    print("+" + "-" * WORLD_WIDTH + "+")
    print("O = Creature    . = Food    X = Predator")


# ---------------- MAIN ----------------

def main():
    creatures = [Creature() for _ in range(POPULATION_SIZE)]

    best_fitness_ever = 0

    for generation in range(1, MAX_GENERATIONS + 1):

        foods = [
            (
                random.randint(0, WORLD_WIDTH - 1),
                random.randint(0, WORLD_HEIGHT - 1)
            )
            for _ in range(FOOD_COUNT)
        ]

        predators = [Predator() for _ in range(PREDATOR_COUNT)]

        total_frames = SECONDS_PER_GENERATION * FPS

        for frame in range(1, total_frames + 1):

            alive = [c for c in creatures if c.alive]

            if alive:
                for creature in alive:
                    creature.sense_and_act(
                        foods,
                        predators,
                        alive
                    )

                for predator in predators:
                    predator.hunt(creatures)

            draw_world(
                creatures,
                foods,
                predators,
                generation,
                frame
            )

            time.sleep(1 / FPS)

            if not any(c.alive for c in creatures):
                break

        best = max(c.fitness() for c in creatures)
        best_fitness_ever = max(best_fitness_ever, best)

        clear_screen()
        print("=" * 45)
        print(f"       GENERATION {generation} FINISHED")
        print("=" * 45)
        print(f"Alive creatures: {sum(c.alive for c in creatures)}")
        print(f"Food eaten: {sum(c.food_eaten for c in creatures)}")
        print(f"Best fitness: {best:.2f}")
        print(f"Best fitness overall: {best_fitness_ever:.2f}")

        if generation < MAX_GENERATIONS:
            print("\nCreating next generation...")
            time.sleep(1.5)

            creatures = create_next_generation(creatures)

    clear_screen()
    print("=" * 45)
    print("       EVOLUTION COMPLETED!")
    print("=" * 45)
    print(f"Total generations: {MAX_GENERATIONS}")
    print(f"Best fitness achieved: {best_fitness_ever:.2f}")
    print("\nSimulation finished.")


if __name__ == "__main__":
    main()