import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds

SIGN_ADD = 11
SIGN_SUB = 12
GRID_SIZE = 10


class PaperWithNumbers:

    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.paper = -1*np.ones(shape=(self.grid_size, self.grid_size))
        self.steps = []

    def make_step(self):
        self.steps.append(self.paper.copy())

    def get_steps(self):
        return self.steps

    def number_to_base(self, n, b=10):
        if n == 0:
            return [0]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return digits[::-1]

    def print_symbol(self, n, x, y):
        self.paper[x][y] = n

    def print_number(self, n, x, y, step_by_step=False):
        n_in_base = self.number_to_base(n)
        for i in range(len(n_in_base)):
            self.paper[x][y - i] = n_in_base[-(i+1)]
            if step_by_step:
                self.make_step()
        return x, y-len(n_in_base)


class AddPlayer:

    def __init__(self, paper):
        self.paper = paper

    def play(self, a, b):

        start = (random.randint(4,6), random.randint(4,6))
        x, y = start

        c = a + b
        self.paper.print_number(a, x, y)
        x, y = x + 1, start[1]
        x, y = self.paper.print_number(b, x, y)
        self.paper.print_symbol(SIGN_ADD, x, y)

        self.paper.make_step()
        x, y = x + 1, start[1]
        x, y = self.paper.print_number(c, x, y, step_by_step=True)


class SubstractPlayer:

    def __init__(self, paper):
        self.paper = paper

    def play(self, a, b):

        start = (random.randint(4,6), random.randint(4,6))
        x, y = start

        c = a - b
        self.paper.print_number(a, x, y)
        x, y = x + 1, start[1]
        x, y = self.paper.print_number(b, x, y)
        self.paper.print_symbol(SIGN_SUB, x, y)

        self.paper.make_step()
        x, y = x + 1, start[1]
        x, y = self.paper.print_number(c, x, y, step_by_step=True)


def generate_dataset(N=1000, grid_size=10):
    xs, ys = [], []
    for i in range(N):
        a = random.randint(100, 500)
        b = random.randint(100, 500)

        paper = PaperWithNumbers(grid_size)
        player = AddPlayer(paper)
        player.play(a, b)
        steps = paper.get_steps()

        for first, second in zip(steps, steps[1:]):
            xs.append(first)
            ys.append(second)

    for i in range(N):
        a = random.randint(500, 900)
        b = random.randint(100, 400)

        paper = PaperWithNumbers(grid_size)
        player = SubstractPlayer(paper)
        player.play(a, b)
        steps = paper.get_steps()

        for first, second in zip(steps, steps[1:]):
            xs.append(first)
            ys.append(second)

    xs = np.array(xs)
    ys = np.array(ys)

    p = np.random.permutation(len(xs))
    xs, ys = xs[p], ys[p]
    return xs, ys
