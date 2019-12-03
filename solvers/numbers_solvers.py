import numpy as np
from .solver import Solver
from utils import number_to_base, is_prime, primes_lt, Symbols as S


class BaseConversionSolver(Solver):

    def _print_number_in_base(self, num, base, **kwargs):
        num_in_base = [int(digit) + 1 for digit in number_to_base(num, base)]
        self.print_symbols_ltr(num_in_base, **kwargs)

    def play(self, problem):
        # TODO: only works with b2=10!!
        n = problem['n']
        b1 = problem['b1']
        b2 = problem['b2']

        num_in_b1 = number_to_base(n, b1)
        num_in_b2 = number_to_base(n, b2)

        self._print_number_in_base(n, b1, attention=True, mark_pos='num',
                                   orientation=1)
        self.make_step()

        self.go_to_mark('start')
        self.move_down()
        self.print_number(b1, orientation=1, attention=1)
        self.print_symbol(S.base_conversion, orientation=1,
                          attention=1, mark_pos='base_conversion_sign')
        self.print_number(b2, orientation=1, attention=1)
        self.make_step()

        self.go_to_mark_range('num')
        self.move_down(2)

        num_in_b1_rev = num_in_b1.copy()
        num_in_b1_rev.reverse()

        for i, digit in enumerate(num_in_b1_rev):
            self.mark_current_pos('s')
            # TODO why 3?
            self.move_left(3)
            self.print_symbol(S.add)
            self.print_number(digit, orientation=1)
            self.print_symbol(S.product, orientation=1)
            self._print_number_in_base(
                b1**i, base=b2, orientation=1)
            self.print_symbol(S.eq, orientation=1)
            self.print_number(b1**i*digit, orientation=1)
            self.make_step()
            self.go_to_mark('s')
            self.move_down()

        self._print_number_in_base(n, b2, attention=True, reset=True)
        self.paper.make_step()
        self.go_to_mark('base_conversion_sign')
        self.move_down()
        self.print_symbol(S.yes)

        self.set_attention_current_pos()
        self.paper.make_step()

        self.set_attention_current_pos(reset=True)
        self.make_step()
        return


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


class DivRemainderSolver(Solver):

    def play(self, problem):
        a = problem['a']
        b = problem['b']

        self.paper._set_position(0, 0)

        self.paper.print_number(a, orientation=1, reset=True)

        self.paper.move_left()
        self.paper.mark_current_pos('margin')
        
        self.paper.move_right()

        self.paper.print_symbol(S.remainder)

        self.paper.print_number(b, orientation=1)
        
        self.paper.make_step()

        for i in range(int((a - a % b) / b)):
            self.paper.go_to_mark('margin')
            self.paper.move_down()
            self.paper.mark_current_pos('margin')
            
            self.paper.print_number(b, orientation=-1)
            self.paper.print_symbol(12, orientation=-1)

            self.paper.go_to_mark('margin')
            self.paper.move_down()
            self.paper.mark_current_pos('margin')
            
            self.paper.make_step(solver='SubtractSolver')

            self.paper.print_number(a - b * (i + 1))


class IsPrimeSolverEasy(Solver):

    def play(self, problem):
        n = problem['a']

        is_prime_ = is_prime(n)

        self.paper.print_number(n, orientation=-1)

        self.mark_current_pos('prime_sign')

        self.paper.print_symbol(S.is_prime, attention=True,
                                preserve_pos=True)

        self.move_down()

        self.make_step()

        if is_prime_:
            self.paper.print_symbol(S.yes, attention=True, reset=True)
        else:
            self.paper.print_symbol(S.no, attention=True, reset=True)

        self.paper.make_step()


class IsPrimeSolverHard(Solver):

    def play(self, problem):
        n = problem['a']
        is_prime_ = is_prime(n)

        self.print_number(n, mark_pos=('number'))

        self.mark_current_pos('prime_sign')

        self.paper.print_symbol(S.is_prime, attention=True,
                                preserve_pos=True)

        self.paper.make_step()

        # ha elég kicsi a szám, akkor tudhatjuk fejből
        if n <= 23:
            self.move_down()
            sign = S.yes if is_prime_ else S.no
            self.paper.print_symbol(sign, attention=True, reset=True)
            self.paper.make_step()
            return

        self.go_to_mark_range('number', end=True)
        # ha ennél nagyobb, akkor ellenőrizzük, hogy vannak-e osztói
        self.move_right(1)
        self.print_symbol(S.sqrt, attention=True, reset=True,
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
            self.print_symbol(S.is_divisible_by,
                              attention=True, reset=True)
            self.set_attention_mark_range('number')
            self.print_number(divisor, orientation=1,
                              attention=True, reset=True)
            self.make_step()
            if n % divisor == 0:
                self.print_symbol(S.yes, attention=True)
                self.go_to_mark('prime_sign')
                self.move_down()
                self.make_step()
                # találtunk osztót, tehát nem prím
                self.print_symbol(S.no, attention=True, reset=True)
                self.make_step()
                return
            else:
                self.print_symbol(S.no, attention=True)
                # self.set_attention(sqrt_pos)
                self.make_step()
            self.go_to_mark('div_sign')
            self.move_down()

        self.go_to_mark('prime_sign')
        self.move_down()
        self.print_symbol(S.yes, attention=True, reset=True)
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
                self.paper.print_number(a, orientation=-1, attention=True)
                self.paper.print_symbols_ltr(
                    [S.div],
                    orientation=-1, attention=True, reset=True)
                self.paper.print_number(factor, orientation=-1, attention=True)
                self.paper.make_step()
                a = a // factor
                j += 1
            i += 1
            factor = primes[i]
            self.paper.make_step()
        self.paper.make_step()


class PlaceValueSolver(Solver):

    def play(self, problem):
        number = problem['number']
        place = problem['place']

        self.paper.print_number(number, orientation=1, reset=True)
        self.print_symbol(S.last_digit)
        self.paper.print_number(place, orientation=1)

        for i in range(1, place + 1):

            self.paper.make_step()

            self.paper._set_position(i, 0)

            for _ in range(i):
                self.paper.print_symbol(0)

            self.paper.print_number(int(str(number)[i:]), orientation=1)
            self.paper.print_symbol(S.last_digit)
            self.paper.print_number(i, orientation=1)


class RoundNumber(Solver):
    pass

def factorize(num):
    factors = []
    while num != 1:
        for i in range(2, num + 1):
            if num % i == 0:
                factors.append(i)
                num = num // i
                break
    return factors

class GCDSolver(Solver):

    def _print_factors(self, mark, number):
        self.go_to_mark_range(mark, end=True)
        self.move_down()
        self.paper.print_symbol(S.factorize, orientation=0)
        self.paper.move_down()
        # TODO: call for other solver!
        factors_a = factorize(number)
        for factor in factors_a:
            self.paper.print_number(factor, orientation=-1, preserve_pos=True)
            self.paper.move_down()

    def print_solution(self):
        self.paper.go_to_mark('answer')
        self.print_number(np.product(self.c_factors), orientation=1)
        self.paper.print_symbol(S.end)

    def _move_mark_down(self, mark):
        self.paper.go_to_mark(mark)
        self.paper.move_down()
        self.paper.mark_current_pos(mark)

    def add_factor_to_c(self, factor, first_factor=True):
        self.paper.go_to_mark('c_current')
        if first_factor:
            self.c_factors = [factor]
        if not first_factor:
            self.paper.print_symbol(S.product, orientation=1)
            self.c_factors.append(factor)
        self.paper.print_number(factor, orientation=1)
        self.paper.mark_current_pos('c_current')

    def play(self, problem):
        a,b = problem['a'], problem['b']
        self.paper.move_down(2)

        self.paper.print_symbol(S.gcd, orientation=-1, mark_pos='gcd')
        self.paper.mark_current_pos('a_current', vertical_offset=2)
        self.paper.print_number(a, orientation=-1, mark_pos='a')
        self.paper.go_to_mark('gcd')
        self.move_right()
        self.paper.print_number(b, orientation=1, mark_pos='b')
        self.paper.mark_current_pos('b_current', vertical_offset=2,
                                    horizontal_offset=-1)

        self._print_factors('a', a)
        self._print_factors('b', b)

        self.paper.go_to_mark('start')
        self.print_symbol(S.gcd)
        self.print_symbol(S.eq)
        self.paper.mark_current_pos('c_current')

        self.paper.go_to_mark('a_current')
        current_factor_a = self.paper.get_number_at_position()
        self.paper.go_to_mark('b_current')
        current_factor_b = self.paper.get_number_at_position()

        first_factor_c = True

        while current_factor_a != 0:
            while current_factor_a != current_factor_b:
                if current_factor_a < current_factor_b:
                    self._move_mark_down('a_current')
                    current_factor_a = self.paper.get_number_at_position()
                if current_factor_a > current_factor_b:
                    self._move_mark_down('b_current')
                    current_factor_b = self.paper.get_number_at_position()
            self.add_factor_to_c(current_factor_a,
                                 first_factor=first_factor_c)
            first_factor_c = False
            self._move_mark_down('a_current')
            current_factor_a = self.paper.get_number_at_position()
            self._move_mark_down('b_current')
            current_factor_b = self.paper.get_number_at_position()
            self.make_step()

        self.print_solution()
        self.paper.set_attention_current_pos()
        self.paper.make_step()
