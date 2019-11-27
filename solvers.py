import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.layers as layers
from common import number_to_base, Symbols as S
from paper import PaperWithNumbers


class Solver:

    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.set_paper()

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
        self.paper.mark_current_pos('a_end', horizontal_offset=-1)
        self.paper.print_symbol(S.product, attention=True)
        self.paper.print_number(b, orientation=1)
        self.paper.mark_current_pos('b_end', horizontal_offset=-1)
        self.paper.make_step()

        b_copy = b
        k = 0
        rs = []

        while b_copy != 0:
            m = b_copy % 10
            r = m * a * 10**k
            rs.append(r)
            # TODO set attention # self.paper.set_attention()
            self.paper.go_to_mark('a_end')
            self.move_down(k+1)
            self.paper.print_number(r, orientation=-1)
            self.paper.make_step()

            b_copy = b_copy // 10
            k += 1

        # TODO set attention
        self.paper.print_symbol(S.add, orientation=-1, attention=True)
        self.paper.make_step()
        self.paper.go_to_mark('a_end')
        self.move_down(k+1)
        final_result = np.sum(rs)
        self.paper.print_number(final_result, orientation=-1)
        self.paper.make_step(solver='AddSolver')

class PaircomparisonSolver(Solver):

    def play(self, problem):
        a, b = problem['a'], problem['b']
        self.paper.print_number(a, orientation=1)
        self.paper.print_symbol(S.greater, attention=True)
        self.paper.print_number(b, orientation=1)
        self.paper.make_step()

        self.go_to_mark('start')
        self.move_down()
        if a > b:
            self.paper.print_symbol(S.yes, attention=True, reset=True)
        else:
            self.paper.print_symbol(S.no, attention=True, reset=True)
        self.paper.make_step()

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

class DivisionSolver(Solver):

    def play(self, problem):
        a, b = problem['a'], problem['b']
        self.paper.print_number(a, orientation=1)
        self.paper.mark_current_pos('a_end', horizontal_offset=-1)
        self.paper.print_symbol(S.div, attention=True)
        self.paper.print_number(b, orientation=1)
        self.paper.mark_current_pos('b_end', horizontal_offset=-1)
        self.paper.print_symbol(S.eq, attention=False)
        self.paper.mark_current_pos('result_start')
        self.paper.make_step()

        k = 1
        a_part = 0
        while a_part < b and a_part < a:
            #TODO set attention
            self.paper.go_to_mark('start')
            self.move_down(k)
            a_part = int(str(a)[:k])
            k+=1
            self.paper.print_number(a_part, orientation=1)
            self.paper.make_step(solver='PaircomparisonSolver')
        if a_part < b:
            self.paper.go_to_mark('result_start')
            self.paper.print_number(0, orientation=1)
            self.paper.make_step(solver='PaircomparisonSolver')



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
    'MultiplySolver',
    'PaircomparisonSolver'
]
