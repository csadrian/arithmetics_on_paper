import numpy as np
from decimal import Decimal
from mathematics_dataset.util.display import Decimal as _Decimal
from .solver import Solver
from utils import Symbols as S


class AddSolver(Solver):

    def play(self, problem):
        a = problem['a']
        b = problem['b']
        base = problem['base']
        
        c = a + b

        self.paper.print_number(a, base, orientation=-1, preserve_pos=True)
        self.move_down()
        self.paper.print_number(b, base, orientation=-1)
        self.paper.print_symbol(S.add, attention=True)
        self.paper.make_step()
        self.go_to_mark('start')
        self.move_down(2)
        self.paper.print_number(c, base, attention=True, orientation=-1, reset=True)
        self.paper.make_step()


class SubtractSolver(Solver):

    def play(self, problem):
        a = problem['a']
        b = problem['b']
        base = problem['base']

        c = a - b

        self.paper.print_number(a, base, orientation=-1, attention=True)
        self.go_to_mark('start')
        self.move_down()
        self.paper.print_number(b, base, orientation=-1, attention=True)
        self.paper.print_symbol(S.sub, orientation=0, attention=True)

        self.paper.make_step()
        self.go_to_mark('start')
        self.move_down(2)
        self.paper.print_number(c, base, attention=True, reset=True)
        self.paper.make_step()


class MultiplySolver(Solver):
    def _decimal_to_int(self, d):
        if isinstance(d, _Decimal):
            d = d._decimal
            nb_decimals = abs(d.as_tuple().exponent)
            d_int = int(d.__mul__(10 ** nb_decimals))
            return d_int, nb_decimals
        else:
            return d, 0

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

    def play(self, problem, verbosity=2):
        if verbosity == 1:
            step_by_step = False
        elif verbosity >= 2:
            step_by_step = True

        a, b = problem.params['p'], problem.params['q']

        nb_dec = 0
        self.paper.go_to_mark('question')
        self.move_right()
        self.paper.print_number(a._decimal, orientation=1)
        self.paper.print_symbol(S.product, orientation=1, attention=True)
        self.paper.print_number(b._decimal, orientation=1)

        a, a_nb_dec = self._decimal_to_int(a)
        b, b_nb_dec = self._decimal_to_int(b)
        nb_dec = a_nb_dec + b_nb_dec

        a, b, sign = self._check_sign(a, b)

        self.paper.go_to_mark('start')
        self.paper.print_number(a, orientation=1, step_by_step=step_by_step)
        self.paper.mark_current_pos('a_end', horizontal_offset=-1)
        self.paper.print_symbol(S.product, orientation=1, step_by_step=step_by_step, attention=True)
        self.paper.print_number(b, orientation=1, step_by_step=step_by_step)
        self.paper.mark_current_pos('b_end', horizontal_offset=-1)

        k_a, k_b = 0, 0
        l = 0
        rs = []
        a_copy, b_copy = a, b

        while b_copy != 0:
            b_digit = b_copy % 10
            a_copy = a
            k_a = 0
            while a_copy != 0:
                a_digit = a_copy % 10
                r = b_digit * a_digit * 10**(k_a + k_b)
                rs.append(r)
                # TODO set attention
                self.paper.go_to_mark('a_end')
                self.move_down(l + 1)
                self.paper.print_number(r, orientation=-1, step_by_step=step_by_step)

                a_copy = a_copy // 10
                k_a += 1
                l += 1
            b_copy = b_copy // 10
            k_b += 1

        # TODO set attention
        rsum = np.sum(rs)
        if len(rs) > 1:
            self.paper.print_symbol(S.add, orientation=-1, attention=True)
            self.paper.make_step()
            self.paper.go_to_mark('a_end')
            self.move_down(l + 1)
            l += 1
            self.paper.print_number(rsum, orientation=-1, step_by_step=step_by_step, solver='AddSolver')

        if nb_dec > 0:
            result = Decimal(str(rsum / 10 **(nb_dec)))
            self.paper.go_to_mark('a_end')
            self.move_down(l + 1)
            self.print_number(result, orientation=-1, step_by_step=step_by_step)
        else:
            result = rsum

        self.go_to_mark('answer')
        if sign == -1:
            self.paper.print_symbol(S.sub, orientation=1)
            self.paper.make_step()
        self.paper.print_number(result, orientation=1, step_by_step=step_by_step)
        self.paper.print_symbol(S.end)
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


class AddSubMultipleSolver(Solver):

    def play(self, problem):
        coefs = problem['coefs']
        ops = problem['ops']

        self.paper.mark_current_pos('start')

        self.paper.print_number(coefs[0], orientation=1, reset=True)
    
        self.paper.move_down()
        self.paper.move_left()
        
        self.paper.mark_current_pos('margin')
        
        total = coefs[0]

        for i in range(len(coefs) - 1):
            
            self.paper.go_to_mark('margin')
            
            self.paper.print_number(coefs[i+1])
            self.print_symbol(ops[i])

            if ops[i] is 11:
                self.make_step(solver='AddSolver')
                total = total + coefs[i+1]
            else:
                self.make_step(solver='SubtractSolver')
                total = total - coefs[i+1]
            
            self.paper.go_to_mark('margin')
            self.paper.move_down()
            self.paper.mark_current_pos('margin')

            self.paper.print_number(total, reset=True, step_by_step=True)
            
            self.paper.go_to_mark('margin')
            self.move_down()
            self.mark_current_pos('margin')
