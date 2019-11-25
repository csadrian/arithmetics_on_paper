import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.layers as layers
from common import number_to_base, Symbols as S

from problems import *

class Step:

    def __init__(self, paper, attention, solver=None):
        self.paper = paper
        self.attention = attention
        self.solver = solver

    def __getitem__(self, key):
        return getattr(self, key)

def print_func(func):
    def inner(*args, **kwargs):
        self = args[0]
        x, y = self._x, self._y
        reserve_pos = kwargs.pop('preserve_pos', False)
        res = func(*args, **kwargs)
        if reserve_pos:
            self._set_position(x, y)
        return res
    return inner

def reset_arg(func):
    def inner(*args, **kwargs):
        self = args[0]
        if kwargs.pop('reset', False):
            self.reset_attention()
        res = func(*args, **kwargs)
        return res
    return inner


class PaperWithNumbers:

    def __init__(self, grid_size, startx=0, starty=5):
        self.shape = (grid_size, grid_size)
        self.paper = -1*np.ones(shape=self.shape, dtype=np.int32)
        self.reset_attention()
        self.steps = []
        self._x = startx
        self._y = starty
        self._marked_cells = dict()
        self._marked_ranges = dict()
        self.mark_current_pos('start')

    def reset_attention(self):
        self.attention = np.zeros(shape=self.shape)

    def make_step(self, solver=None):
        self.steps.append(Step(self.paper.copy(), self.attention.copy(),
                               solver))

    def get_steps(self):
        return self.steps

    def _mark_cell(self, name, pos):
        self._marked_cells[name] = pos

    def _mark_range(self, name, range):
        # range: list of x,y points.
        # sorted ltr!
        self._marked_ranges[name] = sorted(range)

    def _set_position(self, x, y):
        self._x, self._y = x, y

    def go_to_mark(self, name):
        mark = self._marked_cells[name]
        self._x, self._y = mark

    def go_to_mark_range(self, name, end=False):
        """
        end : bool
            if True: go to end of range, otherwise beginning
        """
        mark = self._marked_ranges[name]
        self._x, self._y = mark[-1] if end else mark[0]

    @reset_arg
    def set_attention_mark(self, name):
        self.set_attention([self._marked_cells[name]])

    def remove_attention_mark(self, name):
        self.remove_attention([self._marked_cells[name]])

    @reset_arg
    def set_attention_mark_range(self, name):
        self.set_attention(self._marked_ranges[name])

    def remove_attention_mark_range(self, name):
        self.remove_attention(self._marked_ranges[name])

    def mark_current_pos(self, name):
        self._mark_cell(name, (self._x, self._y))

    @reset_arg
    def set_attention_current_pos(self):
        self.set_attention([(self._x, self._y)])

    @print_func
    @reset_arg
    def print_symbols_ltr(self, ns, attention=False,
                          orientation=1, mark_pos=False,
                          step_by_step=False):
        """
        mark_pos : bool or str
            False/0 if no marking needed, name of the mark otherwise
        """
        x, y = self._x, self._y
        ns = list(ns)
        if orientation < 0:
            ns.reverse()
        for i, symbol in enumerate(ns):
            self.paper[x, y] = symbol
            if attention:
                self.attention[x, y] = 1
            if step_by_step:
                self.make_step()
            y += orientation
        if mark_pos:
            range_ = []
            for i in range(len(ns)):
                range_.append((x, y+orientation*i))
            self._mark_range(mark_pos, range_)
        self._set_position(x, y)

    @print_func
    @reset_arg
    def print_symbol(self, n, attention=False,
                     orientation=1, mark_pos=False):
        x, y = self._x, self._y
        self.paper[x, y] = n
        if attention:
            self.attention[x, y] = 1
        if mark_pos:
            self._mark_cell(mark_pos, (self._x, self._y))
        self._set_position(x, y+orientation)

    @print_func
    @reset_arg
    def print_number(self, n, step_by_step=False, attention=False, 
                     orientation=-1, mark_pos=False):
        x, y = self._x, self._y
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
        if mark_pos:
            res = []
            for i in range(len(n_in_base)):
                res.append((x, y+orientation*i))
            self._mark_range(mark_pos, res)
        self._set_position(x, y+orientation*len(n_in_base))

    @reset_arg
    def set_attention(self, points):
        for (x, y) in points:
            self.attention[x, y] = 1

    def remove_attention(self, points):
        for (x, y) in points:
            self.attention[x, y] = 0

    def move_right(self, n=1):
        self._y = self._y + n

    def move_down(self, n=1):
        self._x = self._x + n 

    def move_left(self, n=1):
        return self.move_right(-1*n)

    def move_up(self, n=1):
        return self.move_down(-1*n)

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

        c = a + b

        self.paper.print_number(a, orientation=-1, preserve_pos=True)
        self.move_down()
        self.paper.print_number(b, orientation=-1)
        self.paper.print_symbol(S.add, attention=True)
        self.paper.make_step()
        self.go_to_mark('start')
        self.move_down(2)
        self.paper.print_number(c, attention=True, orientation=-1, reset=True)
        self.paper.make_step()

class SubtractSolver(Solver):

    def play(self, problem):
        a, b = problem['a'], problem['b']

        c = a - b

        self.paper.print_number(a, orientation=-1, attention=True)
        self.go_to_mark('start')
        self.move_down()
        self.paper.print_number(b, orientation=-1, attention=True)
        self.paper.print_symbol(S.sub, orientation=0, attention=True)

        self.paper.make_step()
        self.go_to_mark('start')
        self.move_down(2)
        self.paper.print_number(c, attention=True, reset=True)
        self.paper.make_step()

class MultiplySolver(Solver):

    def play(self, problem):
        a = problem['a']
        b = problem['b']

        c = a * b

        self.paper.print_number(a, orientation=1)
        self.paper._mark_cell('a_end', (self._x, self._y - 1))
        self.paper.print_symbol(S.product, attention=True)
        self.paper.print_number(b, orientation=1)
        self.paper._mark_cell('b_end', (self._x, self._y - 1))
        self.paper.make_step()

        b_copy = b
        k = 0
        rs = []

        while b_copy != 0:
            m = b_copy % 10
            r = m * a * 10**k
            rs.append(r)
            # TODO set attention # self.paper.set_attention()
            self.paper._set_position(self._marked_cells['a_end'][0], self._marked_cells['a_end'][1])
            self.move_down(k+1)
            self.paper.print_number(r, orientation=-1)
            self.paper.make_step()

            b_copy = b_copy // 10
            k += 1

        # TODO set attention
        self.paper.print_symbol(S.add, orientation=-1, attention=True)
        self.paper.make_step()
        self.paper._set_position(self._marked_cells['a_end'][0], self._marked_cells['a_end'][1])
        self.move_down(k+1)
        final_result = np.sum(rs)
        self.paper.print_number(final_result, orientation=-1)
        self.paper.make_step(solver='AddSolver')

class IsDivisibleBySolver(Solver):

    def play(self, problem):
        a, b = problem['a'], problem['b']

        self.paper.print_number(a)
        self.go_to_mark('start')
        self.move_down()
        self.paper.print_number(b)
        self.paper.print_symbol(S.is_divisible_by)

        self.move_down()
        self.paper.make_step()

        if a % b == 0:
            self.paper.print_symbol(S.yes, attention=True, reset=True)
        else:
            self.paper.print_symbol(S.no, attention=True, reset=True)

        self.paper.make_step()

class FactorizeSolver(Solver):

    def play(self, problem):

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
        a = problem['a']

        self.paper.print_number(a)
        self.paper.print_symbol(S.factorize)

        self.go_to_mark('start')
        self.move_down()

        i = 0
        j = 0

        factor = primes[i]

        self.make_step()

        while a != 1:
            while (a % factor == 0) and (a != 1):
                self.go_to_mark('start')
                self.move_down(j+1)
                self.move_right()
                self.paper.print_symbols_ltr(
                    number_to_base(a) + [S.div] + number_to_base(factor),
                    orientation=-1, attention=True, reset=True)
                self.paper.make_step()
                a = a // factor
                j += 1
            i += 1
            factor = primes[i]
            self.paper.make_step()
        self.paper.make_step()


__all__ = [
    'AddSolver',
    'SubtractSolver',
    'IsDivisibleBySolver',
    'FactorizeSolver',
    'MultiplySolver'
]
