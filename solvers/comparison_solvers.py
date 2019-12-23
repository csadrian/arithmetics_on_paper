import numpy as np
from .solver import Solver
from utils import number_to_base, Symbols as S


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
            result = S.yes
        else:
            self.paper.print_symbol(S.no, attention=True, reset=True)
            result = S.no
        self.paper.make_step()
        self.go_to_mark('answer')
        self.paper.print_symbol(result)
        self.paper.print_symbol(S.end)
        self.paper.make_step()


class SortSolver(Solver):

    def longest_decimal(self, ns):
        lens = [len(number_to_base(n)) for n in ns]
        return np.max(lens)

    def get_number_at_position(self, orientation=-1):
        return int(self.get_word_at_position(orientation))

    def play(self, problem):
        ns = []
        for key in problem.keys():
            if key.startswith('entity'):
                ns.append(problem[key])

        ns = list(np.abs(ns))

        self.print_symbol(S.sort, orientation=0)

        for n in ns:
            self.move_down()
            self.print_number(np.abs(n), orientation=-1, preserve_pos=True)

        self.mark_current_pos('c1_last')

        self.make_step()

        # starting point of second - sorted - column
        self.go_to_mark('start')
        self.move_right(self.longest_decimal(ns)+1)
        self.move_down()
        self.mark_current_pos('c2_first')
        self.make_step()

        self.go_to_mark('start')
        self.move_down()
        self.mark_current_pos('c1')
        self.set_attention_word()
        self.make_step()

        self.go_to_mark('start')
        self.move_down()
        self.mark_current_pos('c1_first')
        while ns:
            min = np.min(ns)
            del ns[ns.index(min)]
            self.go_to_mark('c2_first')
            self.print_number(min, attention=True, reset=True)
            self.go_to_mark('c2_first')
            self.move_down()
            self.mark_current_pos('c2_first')
            self.make_step()
        return
