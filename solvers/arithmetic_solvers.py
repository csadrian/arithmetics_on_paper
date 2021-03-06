import numpy as np
from decimal import Decimal
from mathematics_dataset.util.display import Decimal as _Decimal
from .solver import Solver
from utils import Symbols as S
import math


class AddSolver(Solver):

    def play(self, problem):
        p = problem['p']
        q = problem['q']

        value = p + q

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


class AddOrSubSolver(Solver):
    def _decimal_to_int(self, d):
        if isinstance(d, _Decimal):
            d = d._decimal
            nb_decimals = abs(d.as_tuple().exponent)
            d_int = int(d.__mul__(10 ** nb_decimals))
            return d_int, nb_decimals
        else:
            return d, 0

    def _check_sign(self, n):
        if n < 0:
            sign = -1
            n = n.__neg__()
        else:
            sign = 1
        return n, sign

    def play(self, problem, verbosity=2):
        if verbosity == 1:
            step_by_step = False
        elif verbosity >= 2:
            step_by_step = True

        a, b, problem_type = problem['p'], problem['q'], problem['problem_type']

        if problem_type == 'add':
            problem_sign = 1
        elif problem_type == 'sub':
            problem_sign = -1

        self.paper.go_to_mark('question')
        self.move_right()
        self.paper.print_number(a._decimal, orientation=1)
        if problem_sign == 1:
            self.paper.print_symbol(S.add, orientation=1, attention=True)
        else:
            self.paper.print_symbol(S.sub, orientation=1, attention=True)
        self.paper.print_number(b._decimal, orientation=1)

        a, a_sign = self._check_sign(a)
        b, b_sign = self._check_sign(b)
        b_sign = b_sign * problem_sign
        c_sign = 1

        if (a_sign < b_sign and b > a):
            a, b = b, a
        elif (a_sign > b_sign and a < b):
            a, b = b, a
            c_sign = -1
        elif (a_sign < b_sign and a > b):
            c_sign = -1

        a, a_nb_dec = self._decimal_to_int(a)
        b, b_nb_dec = self._decimal_to_int(b)

        if a_nb_dec > b_nb_dec:
            nb_dec = a_nb_dec
            b = b * 10**(nb_dec - b_nb_dec)
        else:
            nb_dec = b_nb_dec
            a = a * 10**(nb_dec - a_nb_dec)

        self.go_to_mark('start')
        self.paper.print_number(a, orientation=-1, step_by_step=step_by_step)
        self.go_to_mark('start')
        self.move_down()
        self.paper.print_number(b, orientation=-1, step_by_step=step_by_step)

        if a_sign == b_sign:
            self.paper.print_symbol(S.add, orientation=-1, step_by_step=step_by_step)
            c = a + b
            c_sign = a_sign
        else:
            self.paper.print_symbol(S.sub, orientation=-1, step_by_step=step_by_step)
            c = a - b

        self.go_to_mark('start')
        self.move_down(2)
        self.paper.print_number(c, orientation=-1, step_by_step=step_by_step)

        if nb_dec > 0:
            result = Decimal(str(c / 10 **(nb_dec)))
        else:
            result = Decimal(str(c))

        if nb_dec > 0 or c_sign == -1:
            self.paper.go_to_mark('start')
            self.move_down(4)
            self.print_number(result, orientation=-1, step_by_step=step_by_step)
            if c_sign == -1:
                self.paper.print_symbol(S.sub, orientation=-1, step_by_step=step_by_step)

        self.go_to_mark('answer')
        if c_sign == -1:
            self.paper.print_symbol(S.sub, orientation=1, step_by_step=step_by_step)
        self.paper.print_number(result, orientation=1, step_by_step=step_by_step)
        self.paper.print_symbol(S.end)
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

        a, b = problem['p'], problem['q']

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


class MultiplySolver2(Solver):
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

        a, b = problem['p'], problem['q']

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

        if len(str(a)) < len(str(b)):
            a, b = b, a

        self.paper.go_to_mark('start')
        self.paper.print_number(a, orientation=1, step_by_step=step_by_step)
        self.paper.mark_current_pos('a_end', horizontal_offset=-1)
        self.paper.print_symbol(S.product, orientation=1, step_by_step=step_by_step, attention=True)
        self.paper.print_number(b, orientation=1, step_by_step=step_by_step)
        self.paper.mark_current_pos('b_end', horizontal_offset=-1)

        k_a, k_b = 0, 0
        l = 1
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
                self.move_right(4)
                self.paper.print_number(b_digit, orientation=-1, step_by_step=step_by_step)
                self.paper.print_symbol(S.product, orientation=-1, step_by_step=step_by_step)
                self.paper.print_number(a_digit, orientation=-1, step_by_step=step_by_step)
                self.paper.print_symbol(S.eq, orientation=-1, step_by_step=step_by_step)

                self.paper.go_to_mark('a_end')
                self.move_down(l + 1)
                self.move_left(k_a + k_b)
                self.paper.print_number(a_digit*b_digit, orientation=-1, step_by_step=step_by_step)

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
            self.move_down(l + 2)
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


class BasicDivisionSolver(Solver):

    def play(self, problem, verbosity=2):

        step_by_step = self._set_step_by_step(verbosity)

        a, b = problem['p'], problem['q']
        self._print_question(a, S.div, b)

        a, b, sign = self._check_sign(a, b)
        self.paper.go_to_mark('start')
        self.paper.print_number(a, orientation=1, step_by_step=step_by_step)
        self.paper.print_symbol(S.div, orientation=1, step_by_step=step_by_step)
        self.paper.print_number(b, orientation=1, step_by_step=step_by_step)

        self.paper.print_symbol(S.eq)
        self.paper.mark_current_pos('result_next_digit')
        self.paper.make_step()

        self.paper.go_to_mark('start')
        self.paper.mark_current_pos('start', vertical_offset=1)

        dividend, remainder = 0, 0
        result, a_part_str = '', ''
        a_str = str(a)

        while len(a_str) > 0:
            k = 0
            while dividend < b and k <= len(a_str):
                a_part_str = a_str[:k+1]
                k += 1
                dividend = int(str(remainder) + a_part_str)

                self.paper.go_to_mark('start')
                self.paper.print_number(dividend, orientation=1, step_by_step=step_by_step, solver='PaircomparisonSolver')
                self.paper.mark_current_pos('end', horizontal_offset=-1)
                self.paper.go_to_mark('start')
                self.paper.mark_current_pos('start', vertical_offset=1)

                if dividend < b and len(result) > 0:
                    result = result + '0'
                    self.paper.go_to_mark('result_next_digit')
                    self.paper.print_number(0, orientation=1, step_by_step=step_by_step, solver='PaircomparisonSolver')
                    self.paper.mark_current_pos('result_next_digit')

            result_next_digit = dividend // b
            result += str(result_next_digit)
            product = result_next_digit * b
            remainder = dividend - product
            dividend = remainder
            a_str = a_str[k:]
            a_part = ''

            self.paper.go_to_mark('result_next_digit')
            self.paper.print_number(result_next_digit, orientation=1, step_by_step=step_by_step)
            self.paper.mark_current_pos('result_next_digit')

            self.paper.go_to_mark('end')
            self.move_down()
            self.paper.print_number(product, orientation=-1, step_by_step=step_by_step, solver='MultiplySolver')
            self.paper.print_symbol(S.sub, orientation=-1, step_by_step=step_by_step)

            self.paper.go_to_mark('end')
            self.move_down(2)
            self.mark_current_pos('end')
            self.paper.print_number(remainder, orientation=-1, step_by_step=step_by_step, solver='SubSolver')
            self.paper.mark_current_pos('start', vertical_offset=1, horizontal_offset=1)

        final_result = sign * int(result)
        self._print_answer(final_result, step_by_step)

class DivisionSolver(Solver):

    def play(self, problem, verbosity=2):

        step_by_step = self._set_step_by_step(verbosity)

        a, b = problem['p'], problem['q']
        self._print_question(a, S.div, b)

        a, b, sign = self._check_sign(a, b)

        if a%b == 0:
            BasicDivisionSolver.play(self, problem, verbosity) # is this going to work actually?
        else:
            gcd = math.gcd(a, b)
            numerator = a // gcd
            denominatoor = b // gcd
            self.paper.go_to_mark('start')
            self.paper.print_symbol(S.gcd, orientation=1, step_by_step=step_by_step)
            self.paper.print_symbol(S.leftbracket, orientation=1, step_by_step=step_by_step)
            self.paper.print_number(a, orientation=1, step_by_step=step_by_step)
            self.paper.print_symbol(S.comma, orientation=1, step_by_step=step_by_step)
            self.paper.print_number(b, orientation=1, step_by_step=step_by_step)
            self.paper.print_symbol(S.rightbracket, orientation=1, step_by_step=step_by_step)
            self.paper.print_symbol(S.eq)
            self.paper.print_number(gcd, orientation=1, step_by_step=step_by_step, solver='GCDSolver')

            self.paper.go_to_mark('start')
            self.move_down(2)
            self.paper.print_number(a, orientation=1, step_by_step=step_by_step)
            self.paper.print_symbol(S.div, orientation=1, step_by_step=step_by_step)
            self.paper.print_number(gcd, orientation=1, step_by_step=step_by_step)
            self.paper.print_symbol(S.eq)
            self.paper.print_number(numerator, orientation=1, step_by_step=step_by_step, solver='BasicDivisionSolver')

            self.paper.go_to_mark('start')
            self.move_down(4)
            self.paper.print_number(b, orientation=1, step_by_step=step_by_step)
            self.paper.print_symbol(S.div, orientation=1, step_by_step=step_by_step)
            self.paper.print_number(gcd, orientation=1, step_by_step=step_by_step)
            self.paper.print_symbol(S.eq)
            self.paper.print_number(denominator, orientation=1, step_by_step=step_by_step, solver='BasicDivisionSolver')

            self.paper.go_to_mark('start')
            self.move_down(6)
            self.paper.print_number(numerator, orientation=1, step_by_step=step_by_step)
            self.paper.print_symbol(S.div, orientation=1, step_by_step=step_by_step)
            self.paper.print_number(denominator, orientation=1, step_by_step=step_by_step)

            final_result = sign * int(result)
            self._print_answer(final_result, step_by_step)

#WIP
class AddSubMultipleSolver(Solver):

    #Maybe the Op itself has this feature, I didnt check fully.
    def print_op(self, op):
    
        _symbols = dict((y, x) for x, y in S.visual.symbols.iteritems())

        op_text = str(op)

        for letter in op_text:
            if letter.isdigit():
                
                self.paper.print_number(int(letter))

            elif letter in _symbols.keys():
                
                self.paper.print_symbol(_symbols[letter])

            else:
                pass #TypeError

    def play(self, problem):

        op = problem.expression
        value = problem.value
        
        #Need to add header.

        self.paper.mark_current_pos('start')

        print_op(op)
        self.make_step()

        self.paper.go_to_mark('start')

        self.paper.move_down()
        self.paper.set_mark('margin')
    
        children = op.children

        for child in children:
            
            self.paper.print_number(child.value)
            
            if isinstance(child, Constant):
                self.paper.make_step()
            else:
                self.paper.make_step(solver='AddSubMultipleSolver')
