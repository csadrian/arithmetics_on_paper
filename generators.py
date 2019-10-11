import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds

SIGN_ADD = 11
SIGN_SUB = 12
SIGN_IS_PRIME = 13
SIGN_IS_DIVISIBLE_BY = 14
SIGN_FACTORIZE = 15
SIGN_DIV = 16

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


    def print_symbols_ltr(self, ns, x, y):
        for n in ns:
            self.paper[x][y] = n
            y += 1
        return x, y

    def print_symbol(self, n, x, y):
        self.paper[x][y] = n
        return x, y-1

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


class IsPrimePlayer:

    def __init__(self, paper):
        self.paper = paper

    def is_prime(self, x):
        if x < 2:
            return False
        else:
            for n in range(2,x):
                if x % n == 0:
                   return False
            return True

    def play(self, a):

        start = (random.randint(4,6), random.randint(4,6))
        x, y = start

        x, y = self.paper.print_number(a, x, y)
        self.paper.print_symbol(SIGN_IS_PRIME, x, y)
        x, y = x + 1, start[1]
        self.paper.make_step()

        if self.is_prime(a):
            self.paper.print_number(1, x, y)
        else:
            self.paper.print_number(0, x, y)

        self.paper.make_step()



class IsDivisibleByPlayer:

    def __init__(self, limit):
        self.paper = None
        self.limit = limit

    def set_paper(self, paper):
        self.paper = paper

    def generate(self, data_split=None):
        a = random.randint(1, self.limit)
        b = random.randint(1, self.limit)
        a, b = max(a, b), min(a, b)
        self.play(a,b)
        return self.paper.get_steps()

    def play(self, a, b):

        start = (random.randint(4,6), random.randint(4,6))
        x, y = start

        x, y = self.paper.print_number(b, x, y)
        x, y = self.paper.print_symbol(SIGN_IS_DIVISIBLE_BY, x, y)
        x, y = self.paper.print_number(a, x, y)

        x, y = x + 1, start[1]
        self.paper.make_step()

        if a % b == 0:
            self.paper.print_number(1, x, y)
        else:
            self.paper.print_number(0, x, y)

        self.paper.make_step()


class FactorizePlayer:

    def __init__(self, limit):
        self.paper = None
        self.limit = limit

    def set_paper(self, paper):
        self.paper = paper

    def generate(self, data_split=None):
        a = random.randint(1, self.limit)
        self.play(a)
        return self.paper.get_steps()

    def play(self, a):

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

        start = (random.randint(4,6), random.randint(4,6))
        x, y = start

        x, y = self.paper.print_number(a, x, y)
        x, y = self.paper.print_symbol(SIGN_FACTORIZE, x, y)

        #x, y = self.paper.print_number(a, x, y)
        x, y = x + 1, start[1]
        i = 0
        factor = primes[i]
        while a != 1:
            while (a % factor == 0) and (a != 1):
                print(a)
                y = start[1]-3
                x, y = self.paper.print_symbols_ltr(self.paper.number_to_base(a) + [SIGN_DIV] + self.paper.number_to_base(factor), x, y)
                self.paper.make_step()
                x, y = x + 1, start[1]
                a = a // factor
            i += 1
            factor = primes[i]
        self.paper.make_step()


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

    for i in range(N):
        a = random.randint(1, 33)

        paper = PaperWithNumbers(grid_size)
        player = IsPrimePlayer(paper)
        player.play(a)
        steps = paper.get_steps()

        for first, second in zip(steps, steps[1:]):
            xs.append(first)
            ys.append(second)

    generators = [
        IsDivisibleByPlayer(limit=20),
        IsDivisibleByPlayer(limit=50),
        FactorizePlayer(limit=50),
    ]

    for generator in generators:
        for i in range(N):
            paper = PaperWithNumbers(grid_size)
            generator.set_paper(paper)
            steps = generator.generate()
            for first, second in zip(steps, steps[1:]):
                xs.append(first)
                ys.append(second)

    xs = np.array(xs)
    ys = np.array(ys)

    p = np.random.permutation(len(xs))
    xs, ys = xs[p], ys[p]
    return xs, ys
