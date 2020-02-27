from paper import PaperWithNumbers
from utils import Symbols as S


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

    def __getattribute__(self, name, recursive=False):
        if not recursive and name == 'play':
            return self._play
        else:
            return super().__getattribute__(name)

    def _play(self, problem, *args, **kwargs):
        if type(problem) != dict:
            params = problem.params
        else:
            params = problem
        return self.__getattribute__('play', recursive=True)(
            params, *args, **kwargs)

    def play(self, problem):
        raise NotImplementedError

    def _print_question(self, a, problem_symbol, b=None):
        self.paper.go_to_mark('question')
        self.move_right()
        self.paper.print_number(a, orientation=1)
        self.paper.print_symbol(problem_symbol, orientation=1, attention=True)
        if b is not None:
            self.paper.print_number(b, orientation=1)
        self.paper.make_step()

    def _print_answer(self, value, step_by_step):
        self.go_to_mark('answer')
        self.paper.print_number(value, orientation=1, step_by_step=step_by_step)
        self.paper.print_symbol(S.end)
        self.paper.make_step()

    def _set_step_by_step(self, verbosity):
        if verbosity >= 2:
            return True
        else:
            return False

    def _check_sign(self, a, b):
        if a < 0 and b < 0:
            sign = 1
            a, b = a.__neg__(), b.__neg__()
        elif a < 0:
            sign = -1
            a = a.__neg__()
        elif b < 0:
            sign = -1
            b = b.__neg__()
        else:
            sign = 1
        return a, b, sign
