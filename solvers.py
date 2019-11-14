import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.layers as layers

from problems import *
from utils import *

SIGN_ADD = 11
SIGN_SUB = 12
SIGN_IS_PRIME = 13
SIGN_IS_DIVISIBLE_BY = 14
SIGN_FACTORIZE = 15
SIGN_DIV = 16




class PaperWithNumbers:

    def __init__(self, grid_size):
        self.shape = (grid_size, grid_size)
        self.paper = -1*np.ones(shape=self.shape)
        self.reset_attention()
        self.steps = []

    def reset_attention(self):
        self.attention = np.zeros(shape=self.shape)

    def make_step(self):
        self.steps.append({
            'paper': self.paper.copy(),
            'attention': self.attention.copy()
        })

    def get_steps(self):
        return self.steps

    @staticmethod
    def number_to_base(n, b=10):
        if n == 0:
            return [0]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return digits[::-1]

    def print_symbols_ltr(self, ns, x, y):
        for n in ns:
            self.paper[x, y] = n
            y += 1
        return x, y

    def print_symbol(self, n, x, y, attention=False, reset=False):
        self.paper[x, y] = n
        if reset:
            self.reset_attention()
        if attention:
            self.attention[x, y] = 1
        return x, y-1

    def print_number(self, n, x, y, step_by_step=False, attention=False, 
                     axis='vertical', orientation=-1):
        n_in_base = self.number_to_base(n)
        for i in range(len(n_in_base)):
            offset = i * orientation
            if axis == 'vertical':
                self.paper[x, y + offset] = n_in_base[-(i+1)]
            if axis == 'horizontal':
                self.paper[x + offset, y] = n_in_base[-(i+1)]
            if step_by_step:
                self.make_step()
        return x, y-len(n_in_base)

    def set_attention(self, points, reset=True):
        if reset:
            self.reset_attention()
        for (x, y) in points:
            self.attention[x, y] = 1

class Solver:

    def __init__(self, grid_size):
        self.grid_size = grid_size

    def set_paper(self):
        self.paper = PaperWithNumbers(self.grid_size)

    def get_steps(self):
        return self.paper.get_steps()

    def generator(self, problem_generator):
        for problem in problem_generator:
            self.set_paper()
            self.play(problem)
            yield self.get_steps()

    def play(self, problem):
        raise NotImplementedError

class AddSolver(Solver):

    def play(self, problem):
        a = problem['a']
        b = problem['b']
        start = (random.randint(5, 7), random.randint(5, 7))
        x, y = start

        c = a + b
        self.paper.print_number(a, x, y)
        x, y = x + 1, start[1]
        x, y = self.paper.print_number(b, x, y)
        self.paper.print_symbol(SIGN_ADD, x, y, attention=True)
        x, y = x + 1, start[1]
        self.paper.make_step()
        self.paper.set_attention([(x-1, y)])
        self.paper.set_attention([(x-2, y)], reset=False)
        x, y = self.paper.print_number(c, x, y, step_by_step=True)

class IsPrimeSolver(Solver):

    @staticmethod
    def is_prime(x):
        if x < 2:
            return False
        else:
            for n in range(2,x):
                if x % n == 0:
                   return False
            return True

    def play(self, problem):
        a = problem['a']

        start = (random.randint(4,6), random.randint(4,6))
        x, y = start

        x, y = self.paper.print_number(a, x, y)
        self.paper.print_symbol(SIGN_IS_PRIME, x, y, attention=True)
        x, y = x + 1, start[1]
        self.paper.make_step()

        if self.is_prime(a):
            self.paper.print_number(1, x, y)
        else:
            self.paper.print_number(0, x, y)

        self.paper.make_step()

class SubtractSolver(Solver):

    def play(self, problem):
        a, b = problem['a'], problem['b']

        start = (random.randint(4, 6), random.randint(4, 6))
        x, y = start

        c = a - b
        self.paper.print_number(a, x, y)
        x, y = x + 1, start[1]
        x, y = self.paper.print_number(b, x, y)
        self.paper.print_symbol(SIGN_SUB, x, y)

        self.paper.make_step()
        x, y = x + 1, start[1]
        x, y = self.paper.print_number(c, x, y, step_by_step=True)

class IsDivisibleBySolver(Solver):

    def play(self, problem):
        a, b = problem['a'], problem['b']

        start = (random.randint(4, 6), random.randint(4, 6))
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

class FactorizeSolver(Solver):

    def __init__(self, limit):
        self.paper = None
        self.limit = limit

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
                y = start[1]-3
                x, y = self.paper.print_symbols_ltr(self.paper.number_to_base(a) + [SIGN_DIV] + self.paper.number_to_base(factor), x, y)
                self.paper.make_step()
                x, y = x + 1, start[1]
                a = a // factor
            i += 1
            factor = primes[i]
        self.paper.make_step()

__all__ = [
    'IsPrimeSolver',
    'AddSolver',
    'SubtractSolver',
    'IsDivisibleBySolver',
    'FactorizeSolver'
]
