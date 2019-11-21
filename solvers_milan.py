import numpy as np
from solvers import Solver, AddSolver, SIGN_IS_PRIME, SIGN_ADD,\
    SIGN_YES, SIGN_NO, SIGN_SQRT, IsPrimeSolver, SIGN_IS_DIVISIBLE_BY,\
    SIGN_BASE_CONVERSION, SIGN_PRODUCT, SIGN_EQ

from common import number_to_base, is_prime, primes_lt

class IsPrimeSolverEasy(Solver):

    def play(self, problem):
        n = problem['a']

        is_prime = IsPrimeSolver.is_prime(n)

        start = (5, 5)
        x, y = start
        
        x, y = self.paper.print_number(n, x, y)

        isprimesignpos = x, y
        self.paper.print_symbol(SIGN_IS_PRIME, *isprimesignpos,
                                attention=True)

        x, y = x + 1, start[1]
        self.paper.make_step()

        if is_prime:
            self.paper.print_symbol(SIGN_YES, x, y, attention=True, reset=True)
        else:
            self.paper.print_symbol(SIGN_NO, x, y, attention=True, reset=True)

        self.paper.make_step()

class IsPrimeSolverHard(Solver):

    def play(self, problem):
        start = (1, 4)

        n = problem['a']
        is_prime = IsPrimeSolver.is_prime(n)

        x, y = start

        (x, y), number_pos = self.paper.print_number(
            n, x, y, return_full_pos=True)

        isprimesignpos = x, y

        self.paper.print_symbol(SIGN_IS_PRIME, *isprimesignpos,
                                attention=True)

        self.paper.make_step()

        # ha elég kicsi a szám, akkor tudhatjuk fejből
        if n <= 23:
            x, y = self.move_down(x, y)
            sign = SIGN_YES if is_prime else SIGN_NO
            self.paper.print_symbol(sign, x, y, attention=True, reset=True)
            self.paper.make_step()
            return


        # ha ennél nagyobb, akkor ellenőrizzük, hogy vannak-e osztói
        x, y = self.move_right(*start, 1)
        self.print_symbol(SIGN_SQRT, x, y, attention=True, reset=True)

        self.paper.make_step()

        x, y = self.move_right(x, y, 1)

        # +2: mert csak becsüljük a gyököt, inkább legyen biztos.
        # kell?
        sqrt_n = int(np.sqrt(n)) + 2

        _, sqrt_pos = self.print_number(
            sqrt_n, x, y, orientation=1, return_full_pos=True)

        self.paper.make_step()

        possible_divisors = primes_lt(sqrt_n)

        x, y = self.move_down(x, y)
        x, y = self.move_left(x, y)


        for divisor in possible_divisors:
            div_sign_pos = x, y
            self.print_symbol(SIGN_IS_DIVISIBLE_BY, *div_sign_pos,
                              attention=True, reset=True)
            self.set_attention(number_pos)
            x, y = self.move_right(*div_sign_pos)
            x, y = self.print_number(divisor, x, y, orientation=1,
                                     attention=True, reset=True)
            self.make_step()
            if n % divisor == 0:
                self.print_symbol(SIGN_YES, x, y, attention=True)
                x, y = self.move_down(*isprimesignpos)
                self.make_step()
                # találtunk osztót, tehát nem prím
                self.print_symbol(SIGN_NO, x, y, attention=True, reset=True)
                self.make_step()
                return
            else:
                self.print_symbol(SIGN_NO, x, y, attention=True)
                self.set_attention(sqrt_pos)
                self.make_step()
            x, y = self.move_down(*div_sign_pos)

        x, y = self.move_down(*isprimesignpos)
        self.print_symbol(SIGN_YES, x, y, attention=True, reset=True)
        self.paper.make_step()

        self.set_attention([(x, y)], reset=False)
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
