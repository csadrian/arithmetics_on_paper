from .solver import Solver
from common import Symbols as S


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
