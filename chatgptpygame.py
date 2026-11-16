import pygame
import numpy as np
import random
import sys

# ---------------- تنظیمات ----------------

WIDTH, HEIGHT = 900, 650

POPULATION_SIZE = 70
FOOD_COUNT = 20
PREDATOR_COUNT = 4
MAX_GENERATIONS = 20
MUTATION_RATE = 0.15

# رنگ‌ها
BG_COLOR = (15, 15, 20)
FOOD_COLOR = (0, 255, 100)
CREATURE_COLOR = (255, 60, 60)
PREDATOR_COLOR = (180, 50, 255)
ELITE_COLOR = (255, 255, 255)
TEXT_COLOR = (255, 255, 255)
INFO_BG_COLOR = (30, 30, 40)


# ---------------- شبکه عصبی ----------------

class NeuralNetwork:

    def __init__(self, input_size=6, hidden_size=8,
                 output_size=2, weights=None):

        if weights is None:
            self.w1 = np.random.randn(input_size, hidden_size) * 0.8
            self.w2 = np.random.randn(hidden_size, output_size) * 0.8
        else:
            self.w1 = weights[0].copy()
            self.w2 = weights[1].copy()

    def get_weights(self):
        return self.w1.copy(), self.w2.copy()

    def forward(self, inputs):
        hidden = np.dot(inputs, self.w1)
        hidden = np.maximum(0, hidden)

        output = np.tanh(np.dot(hidden, self.w2))
        return output


# ---------------- موجودات ----------------

class Creature:

    def __init__(self, x, y, brain=None):

        self.x = x
        self.y = y

        self.energy = 100
        self.food_eaten = 0
        self.survival_time = 0

        self.alive = True

        self.brain = brain if brain is not None else NeuralNetwork()

        self.fitness = 0

    def sense_and_act(self, foods, predators, friends):

        if not self.alive:
            return

        self.survival_time += 1

        # حسگر غذا
        f_dx, f_dy = 0, 0

        if foods:
            nearest_food = min(
                foods,
                key=lambda f: (f[0] - self.x)**2 + (f[1] - self.y)**2
            )

            f_dx = (nearest_food[0] - self.x) / WIDTH
            f_dy = (nearest_food[1] - self.y) / HEIGHT

        # حسگر شکارچی
        p_dx, p_dy = 0, 0

        if predators:
            nearest_pred = min(
                predators,
                key=lambda p: (p.x - self.x)**2 + (p.y - self.y)**2
            )

            p_dx = (nearest_pred.x - self.x) / WIDTH
            p_dy = (nearest_pred.y - self.y) / HEIGHT

        # حسگر دوست
        fr_dx, fr_dy = 0, 0

        if friends:
            nearest_friend = min(
                friends,
                key=lambda fr: (fr.x - self.x)**2 + (fr.y - self.y)**2
            )

            fr_dx = (nearest_friend.x - self.x) / WIDTH
            fr_dy = (nearest_friend.y - self.y) / HEIGHT

        # ورودی شبکه عصبی
        inputs = np.array([
            f_dx, f_dy,
            p_dx, p_dy,
            fr_dx, fr_dy
        ])

        output = self.brain.forward(inputs)

        # حرکت موجود
        self.x += output[0] * 4.0
        self.y += output[1] * 4.0

        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))

        # مصرف انرژی
        self.energy -= 0.15

        if self.energy <= 0:
            self.alive = False


# ---------------- شکارچی ----------------

class Predator:

    def __init__(self):

        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)

        self.speed = 1.5

    def hunt(self, creatures):

        if not creatures:
            return

        target = min(
            creatures,
            key=lambda c: (c.x - self.x)**2 + (c.y - self.y)**2
        )

        dx = target.x - self.x
        dy = target.y - self.y

        dist = max(1, np.hypot(dx, dy))

        self.x += (dx / dist) * self.speed
        self.y += (dy / dist) * self.speed

        # شکار موجودات نزدیک
        for creature in creatures:

            distance_squared = (
                (creature.x - self.x)**2 +
                (creature.y - self.y)**2
            )

            if distance_squared < 180:
                creature.alive = False


# ---------------- ترکیب ژنتیکی ----------------

def crossover(parent1, parent2):

    p1_w1, p1_w2 = parent1.brain.get_weights()
    p2_w1, p2_w2 = parent2.brain.get_weights()

    mask1 = np.random.rand(*p1_w1.shape) > 0.5
    mask2 = np.random.rand(*p1_w2.shape) > 0.5

    child_w1 = np.where(mask1, p1_w1, p2_w1)
    child_w2 = np.where(mask2, p1_w2, p2_w2)

    return NeuralNetwork(weights=(child_w1, child_w2))


# ---------------- جهش ژنتیکی ----------------

def mutate(brain):

    w1, w2 = brain.get_weights()

    if random.random() < MUTATION_RATE:
        w1 += np.random.randn(*w1.shape) * 0.3

    if random.random() < MUTATION_RATE:
        w2 += np.random.randn(*w2.shape) * 0.3

    return NeuralNetwork(weights=(w1, w2))


# ---------------- ساخت نسل جدید ----------------

def create_next_generation(population):

    for creature in population:

        creature.fitness = (
            creature.food_eaten * 100
            + creature.survival_time * 5
        )

    population.sort(
        key=lambda c: c.fitness,
        reverse=True
    )

    top_performers = population[
        :int(POPULATION_SIZE * 0.4)
    ]

    new_population = []
    elites = []

    # انتقال ۵ موجود نخبه
    for parent in top_performers[:5]:

        elite_brain = NeuralNetwork(
            weights=parent.brain.get_weights()
        )

        elite = Creature(
            random.randint(0, WIDTH),
            random.randint(0, HEIGHT),
            brain=elite_brain
        )

        elites.append(elite)
        new_population.append(elite)

    # تولید فرزندان
    while len(new_population) < POPULATION_SIZE:

        p1 = random.choice(top_performers)
        p2 = random.choice(top_performers)

        child_brain = crossover(p1, p2)
        child_brain = mutate(child_brain)

        child = Creature(
            random.randint(0, WIDTH),
            random.randint(0, HEIGHT),
            brain=child_brain
        )

        new_population.append(child)

    return new_population, elites


# ---------------- برنامه اصلی ----------------

def main():

    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    pygame.display.set_caption(
        "Neuroevolution - High Competition"
    )

    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont("Tahoma", 24, bold=True)
    font_small = pygame.font.SysFont("Tahoma", 18, bold=True)

    generation = 1
    game_over = False

    # ساخت نسل اول
    population = [
        Creature(
            random.randint(0, WIDTH),
            random.randint(0, HEIGHT)
        )
        for _ in range(POPULATION_SIZE)
    ]

    foods = [
        [random.randint(0, WIDTH), random.randint(0, HEIGHT)]
        for _ in range(FOOD_COUNT)
    ]

    predators = [
        Predator() for _ in range(PREDATOR_COUNT)
    ]

    elites = []

    running = True

    while running:

        # رویدادها
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            # رفتن به نسل بعد با Space
            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_SPACE
                and not game_over
            ):

                population, elites = create_next_generation(population)

                foods = [
                    [random.randint(0, WIDTH), random.randint(0, HEIGHT)]
                    for _ in range(FOOD_COUNT)
                ]

                predators = [
                    Predator() for _ in range(PREDATOR_COUNT)
                ]

                generation += 1

                if generation > MAX_GENERATIONS:
                    game_over = True
                    generation = MAX_GENERATIONS

        screen.fill(BG_COLOR)

        if not game_over:

            # رسم غذاها
            for food in foods:

                pygame.draw.circle(
                    screen,
                    FOOD_COLOR,
                    (int(food[0]), int(food[1])),
                    6
                )

            alive_creatures = [
                c for c in population if c.alive
            ]

            # حرکت شکارچی‌ها
            for predator in predators:

                predator.hunt(alive_creatures)

                pygame.draw.circle(
                    screen,
                    (80, 0, 100),
                    (int(predator.x), int(predator.y)),
                    15,
                    2
                )

                pygame.draw.circle(
                    screen,
                    PREDATOR_COLOR,
                    (int(predator.x), int(predator.y)),
                    10
                )

            # حرکت موجودات
            for creature in alive_creatures:

                friends = [
                    c for c in alive_creatures
                    if c is not creature
                ]

                creature.sense_and_act(
                    foods,
                    predators,
                    friends
                )

                if creature.alive:

                    if creature in elites:
                        pygame.draw.circle(
                            screen,
                            ELITE_COLOR,
                            (int(creature.x), int(creature.y)),
                            9,
                            2
                        )
                    else:
                        pygame.draw.circle(
                            screen,
                            CREATURE_COLOR,
                            (int(creature.x), int(creature.y)),
                            7
                        )

                    # خوردن غذا
                    for food in foods[:]:

                        distance_squared = (
                            (creature.x - food[0])**2 +
                            (creature.y - food[1])**2
                        )

                        if distance_squared < 120:

                            foods.remove(food)

                            creature.energy = min(
                                150,
                                creature.energy + 40
                            )

                            creature.food_eaten += 1

                            foods.append([
                                random.randint(0, WIDTH),
                                random.randint(0, HEIGHT)
                            ])

        # ---------------- HUD ----------------

        pygame.draw.rect(
            screen,
            INFO_BG_COLOR,
            (0, 0, WIDTH, 90)
        )

        screen.blit(
            font_large.render(
                f"Generation: {generation} / {MAX_GENERATIONS}",
                True,
                TEXT_COLOR
            ),
            (20, 15)
        )

        alive_count = sum(
            1 for c in population if c.alive
        )

        screen.blit(
            font_small.render(
                f"Alive: {alive_count} / {POPULATION_SIZE}",
                True,
                TEXT_COLOR
            ),
            (20, 55)
        )

        if not game_over:

            screen.blit(
                font_small.render(
                    "Press SPACE to advance to next generation",
                    True,
                    (255, 215, 0)
                ),
                (400, 55)
            )

        # پایان شبیه‌سازی
        if game_over:

            victory_text = font_large.render(
                "EVOLUTION COMPLETE! 20 Generations Passed.",
                True,
                (0, 255, 0)
            )

            screen.blit(
                victory_text,
                (
                    WIDTH // 2 - victory_text.get_width() // 2,
                    HEIGHT // 2
                )
            )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


# اجرای برنامه
if __name__ == "__main__":
    main()