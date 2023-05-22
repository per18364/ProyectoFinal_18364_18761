import torch
import random
import numpy as np
from snake import SnakeGameAI, Direction, Point
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agente:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0

    def get_state(self, game):
        pass

    def remember(self, state, accion, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self):
        pass

    def get_accion(self, state):
        pass


def train():
    pass


if __name__ == '__main__':
    train()
