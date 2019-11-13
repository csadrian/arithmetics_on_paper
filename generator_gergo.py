import numpy as np
import random
import tensorflow as tf
#import tensorflow.keras.layers as layers
#import tensorflow_datasets as tfds

SIGN_ADD = 11
SIGN_SUB = 12
SIGN_IS_PRIME = 13
SIGN_IS_DIVISIBLE_BY = 14
SIGN_FACTORIZE = 15
SIGN_DIV = 16

UNKNWN_X = 17

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

    def print_number(self, n, x, y, direction=1, step_by_step=False):
        n_in_base = self.number_to_base(n)
        for i in range(len(n_in_base)):
            self.paper[x][y + direction * i] = n_in_base[::direction][i]
            if step_by_step:
                self.make_step()
        return x, y + direction * len(n_in_base)


class Solver:

    def __init__(self, grid_size, parameter_option, custom_paremeter = None):
        self.paper = PaperWithNumbers(grid_size)

        self.parameter_set = {
                'add1' : [[0, 0], [0, 0]]
                'linear_1d': [[-10, 10], [-10, 10]]
                }

    def set_paper(self, paper):
        self.paper = paper

    def get_steps(self):
        return self.paper.get_steps()

    def generate(self, entries = 1, verbosity = 0, generator_mode = False):
        xs = np.array([])
        for _ in range(entries):
            xs = np.append(xs, play(verbosity))
        return xs

    def polynomial_generator(deg, limits):
        poly = []
        for _ in range(deg + 1):
            poly.append(random.randint(limits[_][0], limits[_][1]))
        return poly

class AddSolver(Solver):

    def play(self, verbosity):

        print('AddSolver')

class SubSolver(Solver):

    def play(self, verbosity):
        
        print('SubSolver')

class linear_1d(Solver):

    def play(self, verbosity):
        
        limits = self.paramter_set['linear_1d']

        left_side = poly_gen(1)
        right_side = poly_gen(1)

def generate(solvers):
    for solver in solvers:
        solver.generate()
