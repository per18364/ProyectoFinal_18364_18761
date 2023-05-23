import torch
import random
import numpy as np
from snake import SnakeGameAI, Direction, Point
from collections import deque
from plotter import plot
from ai import LinearQNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agente:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.8
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, 256, 3) #input=11*, hidden, output=3* acciones      *valores necesarios
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        cabeza = game.snake[0]

        pointL = Point(cabeza.x - 20, cabeza.y)
        pointR = Point(cabeza.x + 20, cabeza.y)
        pointU = Point(cabeza.x, cabeza.y - 20)
        pointD = Point(cabeza.x, cabeza.y + 20)

        dirL = game.direccion == Direction.LEFT
        dirR = game.direccion == Direction.RIGHT
        dirU = game.direccion == Direction.UP
        dirD = game.direccion == Direction.DOWN

        state = [
            #Peligro adelante
            (dirR and game.is_collision(pointR)) or
            (dirL and game.is_collision(pointL)) or
            (dirU and game.is_collision(pointU)) or
            (dirD and game.is_collision(pointD)),

            #Peligro a la derecha
            (dirU and game.is_collision(pointR)) or
            (dirD and game.is_collision(pointL)) or
            (dirL and game.is_collision(pointU)) or
            (dirR and game.is_collision(pointD)),

            #Peligro a la izquierda
            (dirD and game.is_collision(pointR)) or
            (dirU and game.is_collision(pointL)) or
            (dirR and game.is_collision(pointU)) or
            (dirL and game.is_collision(pointD)),

            #Movimiento
            dirL,
            dirR,
            dirU,
            dirD,

            #Posicion comida
            game.comida.x < game.cabeza.x, #comida izquierda
            game.comida.x > game.cabeza.x, #comida derecha
            game.comida.y < game.cabeza.y, #comida arriba
            game.comida.y > game.cabeza.y #comida abajo
        ]
        return np.array(state, dtype=int)

    def remember(self, state, accion, reward, nextState, done):
        self.memory.append((state, accion, reward, nextState, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            miniSample = random.sample(self.memory, BATCH_SIZE)
        else:
            miniSample = self.memory

        states, acciones, rewards, nextStates, dones = zip(*miniSample)
        self.trainer.train_step(states, acciones, rewards, nextStates, dones)

    def train_short_memory(self, state, accion, reward, nextState, done):
        self.trainer.train_step(state, accion, reward, nextState, done)

    def get_accion(self, state):
        self.epsilon = 80 - self.n_games
        movFinal = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            movFinal[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            movFinal[move] = 1
        return movFinal


def train():
    scores = []
    meanScores = []
    totalScore = 0
    record = 0
    agente = Agente()
    game = SnakeGameAI()
    while True:
        oldState = agente.get_state(game)
        finalMove = agente.get_accion(oldState)

        #movimiento y cambio de estados
        reward, done, score = game.play_step(finalMove)
        newState = agente.get_state(game)

        #entrenamiento de corto plazo
        agente.train_short_memory(oldState, finalMove, reward, newState, done)

        #remember
        agente.remember(oldState, finalMove, reward, newState, done)

        if done:
            #entrenamiento de largo plazo
            game.reset()
            agente.n_games +=1
            agente.train_long_memory()

            if score > record:
                record = score
                #agente.model.save()
            
            print('Game: ', agente.n_games, 'Score: ', score, 'Record: ', record)

            scores.append(score)
            totalScore += score
            mean_score = totalScore / agente.n_games
            meanScores.append(mean_score)
            plot(scores, meanScores)

if __name__ == '__main__':
    train()
