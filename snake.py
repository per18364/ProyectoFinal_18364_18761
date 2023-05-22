import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 20


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direccion = Direction.RIGHT

        self.cabeza = Point(self.w/2, self.h/2)
        self.snake = [self.cabeza,
                      Point(self.cabeza.x-BLOCK_SIZE, self.cabeza.y),
                      Point(self.cabeza.x-(2*BLOCK_SIZE), self.cabeza.y)]

        self.score = 0
        self.comida = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.comida = Point(x, y)
        if self.comida in self.snake:
            self._place_food()

    def play_step(self, accion):
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(accion)
        self.snake.insert(0, self.cabeza)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.cabeza == self.comida:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def is_collision(self, punto=None):
        if punto is None:
            punto = self.cabeza

        if punto.x > self.w - BLOCK_SIZE or punto.x < 0 or punto.y > self.h - BLOCK_SIZE or punto.y < 0:
            return True

        if punto in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(
                pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2,
                             pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.comida.x, self.comida.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Puntos: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, accion):
        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        index = clock_wise.index(self.direccion)

        if np.array_equal(accion, [1, 0, 0]):
            new_dir = clock_wise[index]
        elif np.array_equal(accion, [0, 1, 0]):
            next_index = (index + 1) % 4
            new_dir = clock_wise[next_index]
        else:
            next_index = (index - 1) % 4
            new_dir = clock_wise[next_index]

        self.direccion = new_dir
        x = self.cabeza.x
        y = self.cabeza.y
        if self.direccion == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direccion == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direccion == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direccion == Direction.UP:
            y -= BLOCK_SIZE

        self.cabeza = Point(x, y)
