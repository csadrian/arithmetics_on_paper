from .solver import Solver
from common import Symbols as S


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
