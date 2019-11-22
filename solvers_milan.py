import numpy as np
from solvers import Solver, AddSolver, SIGN_IS_PRIME, SIGN_ADD,\
    SIGN_YES, SIGN_NO, SIGN_SQRT, IsPrimeSolver, SIGN_IS_DIVISIBLE_BY,\
    SIGN_BASE_CONVERSION, SIGN_PRODUCT, SIGN_EQ

from common import number_to_base, is_prime, primes_lt

class IsPrimeSolverEasy(Solver):

    def play(self, problem):
        n = problem['a']

        is_prime = IsPrimeSolver.is_prime(n)

        self.paper.print_number(n, orientation=-1)

        self.mark_current_pos('prime_sign')

        self.paper.print_symbol(SIGN_IS_PRIME, attention=True,
                                preserve_pos=True)

        self.move_down()

        self.make_step()

        if is_prime:
            self.paper.print_symbol(SIGN_YES, attention=True, reset=True)
        else:
            self.paper.print_symbol(SIGN_NO, attention=True, reset=True)

        self.paper.make_step()

class IsPrimeSolverHard(Solver):

    def play(self, problem):
        n = problem['a']
        is_prime = IsPrimeSolver.is_prime(n)

        self.print_number(n, mark_pos=('number'))

        self.mark_current_pos('prime_sign')

        self.paper.print_symbol(SIGN_IS_PRIME, attention=True,
                                preserve_pos=True)

        self.paper.make_step()

        # ha elég kicsi a szám, akkor tudhatjuk fejből
        if n <= 23:
            self.move_down()
            sign = SIGN_YES if is_prime else SIGN_NO
            self.paper.print_symbol(sign, attention=True, reset=True)
            self.paper.make_step()
            return

        self.go_to_mark_range('number', end=True)
        # ha ennél nagyobb, akkor ellenőrizzük, hogy vannak-e osztói
        self.move_right(1)
        self.print_symbol(SIGN_SQRT, attention=True, reset=True,
                          orientation=1)
        self.paper.make_step()

        # +2: mert csak becsüljük a gyököt, inkább legyen biztos.
        # kell?
        sqrt_n = int(np.sqrt(n)) + 2

        self.print_number(sqrt_n, orientation=1, mark_pos='sqrt',
                          preserve_pos=True)

        self.paper.make_step()

        possible_divisors = primes_lt(sqrt_n)

        self.move_down()
        self.move_left()

        for divisor in possible_divisors:
            self.mark_current_pos('div_sign')
            self.print_symbol(SIGN_IS_DIVISIBLE_BY,
                              attention=True, reset=True)
            self.set_attention_mark_range('number')
            self.print_number(divisor, orientation=1,
                              attention=True, reset=True)
            self.make_step()
            if n % divisor == 0:
                self.print_symbol(SIGN_YES, attention=True)
                self.go_to_mark('prime_sign')
                self.move_down()
                self.make_step()
                # találtunk osztót, tehát nem prím
                self.print_symbol(SIGN_NO, attention=True, reset=True)
                self.make_step()
                return
            else:
                self.print_symbol(SIGN_NO, attention=True)
                # self.set_attention(sqrt_pos)
                self.make_step()
            self.go_to_mark('div_sign')
            self.move_down()

        self.go_to_mark('prime_sign')
        self.move_down()
        self.print_symbol(SIGN_YES, attention=True, reset=True)
        self.paper.make_step()

class RoundNumber(Solver):
    pass

class BaseConversionSolver(Solver):

    def play(self, problem):
        # TODO: only works with b2=10!!
        n = problem['n']
        b1 = problem['b1']
        b2 = problem['b2']

        start = (1, 2)
        # printing out the basic information

        num_in_b1 = number_to_base(n, b1)
        num_in_b2 = number_to_base(n, b2)

        (x, y), numpos = self.print_symbols_ltr(
            num_in_b1, *start, attention=True, return_full_pos=True)
        x, y = self.move_down(*numpos[0])
        x, y = self.print_number(b1, x, y, orientation=1, attention=1)

        pos_sign_base_conversion = x, y
        x, y = self.print_symbol(SIGN_BASE_CONVERSION, x, y, orientation=1,
                                 attention=1)
        x, y = self.print_number(b2, x, y, orientation=1, attention=1)
        self.make_step()

        x, y = self.move_down(*numpos[-1], 2)

        num_in_by_rev = num_in_b1.copy()
        num_in_by_rev.reverse()

        x, y = self.move_right(x, y, 2)
        for i, digit in enumerate(num_in_by_rev):
            sx, sy = x, y
            self.print_symbol(SIGN_ADD, *self.move_left(x, y))
            x, y = self.print_number(digit, x, y, orientation=1)
            x, y = self.print_symbol(SIGN_PRODUCT, x, y, orientation=1)

            x, y = self.print_symbols_ltr(
                number_to_base(b1**i, b=b2), x, y, orientation=1)
            x, y = self.print_symbol(SIGN_EQ, x, y, orientation=1)
            x, y = self.print_number(b1**i*digit, x, y, orientation=1)
            self.make_step()
            x, y = self.move_down(sx, sy)
        self.print_symbols_ltr(num_in_b2, x, y, attention=True,
                               reset=True)
        self.paper.make_step()
        x, y = self.move_down(*pos_sign_base_conversion)
        self.print_symbol(SIGN_YES, x, y)

        self.set_attention([(x, y)], reset=False)
        self.paper.make_step()
