import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.layers as layers
from common import number_to_base

from problems import *

SIGN_ADD = 11
SIGN_SUB = 12
SIGN_IS_PRIME = 13
SIGN_IS_DIVISIBLE_BY = 14
SIGN_FACTORIZE = 15
SIGN_DIV = 16
SIGN_YES = 17
SIGN_NO = 18
SIGN_SQRT = 19
SIGN_BASE_CONVERSION = 20
SIGN_PRODUCT = 21
SIGN_EQ = 22

class Step:

    def __init__(self, paper, attention, solver=None):
        self.paper = paper
        self.attention = attention
        self.solver = solver

    def __getitem__(self, key):
        if key is 'paper':
            return self.paper
        elif key is 'attention':
            return self.solver
        elif key is 'solver':
            return self.solver

class PaperWithNumbers:

    def __init__(self, grid_size):
        self.shape = (grid_size, grid_size)
        self.paper = -1*np.ones(shape=self.shape)
        self.reset_attention()
        self.steps = []

    def reset_attention(self):
        self.attention = np.zeros(shape=self.shape)

    def make_step(self, solver=None):
        self.steps.append(Step(self.paper.copy(), self.attention, solver))

    def get_steps(self):
        return self.steps

    def print_symbols_ltr(self, ns, x, y, attention=False, reset=False,
                          orientation=1, return_full_pos=False,
                          step_by_step=False):
        ns = list(ns)
        if reset:
            self.reset_attention()
        if orientation > 0:
            ns.reverse()
        for i in range(len(ns)):
            offset = i * orientation
            self.paper[x, y + offset] = ns[-(i+1)]
            if attention:
                self.attention[x, y + offset] = 1
            if step_by_step:
                self.make_step()
        if return_full_pos:
            res = []
            for i in range(len(ns)):
                res.append((x, y+orientation*i))
            return (x, y+orientation*len(ns)), res
        else:
            return x, y+orientation*len(ns)

    def print_symbol(self, n, x, y, attention=False, reset=False,
                     orientation=-1):
        self.paper[x, y] = n
        if reset:
            self.reset_attention()
        if attention:
            self.attention[x, y] = 1
        return x, y+orientation

    def print_number(self, n, x, y, step_by_step=False, attention=False, 
                     orientation=-1, return_full_pos=False, reset=False):
        if reset:
            self.reset_attention()
        n_in_base = number_to_base(n)
        if orientation > 0:
            n_in_base.reverse()
        for i in range(len(n_in_base)):
            offset = i * orientation
            self.paper[x, y + offset] = n_in_base[-(i+1)]
            if attention:
                self.attention[x, y + offset] = 1
            if step_by_step:
                self.make_step()
        if return_full_pos:
            res = []
            for i in range(len(n_in_base)):
                res.append((x, y+orientation*i))
            return (x, y+orientation*len(n_in_base)), res
        else:
            return x, y+orientation*len(n_in_base)

    def set_attention(self, points, reset=False):
        if reset:
            self.reset_attention()
        for (x, y) in points:
            self.attention[x, y] = 1

    def remove_attention(self, points):
        for (x, y) in points:
            self.attention[x, y] = 0

    def move_right(self, x, y, n=1):
        return x, y+n

    def move_down(self, x, y, n=1):
        return x+n, y

    def move_left(self, x, y, n=1):
        return self.move_right(x, y, -1*n)

    def move_up(self, x, y, n=1):
        return self.move_down(x, y, -1*n)

class Solver:

    def __init__(self, grid_size):
        self.grid_size = grid_size

    def __getattr__(self, attr):
        return getattr(self.paper, attr)

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
                x, y = self.paper.print_symbols_ltr(number_to_base(a) + [SIGN_DIV] + number_to_base(factor), x, y)
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
