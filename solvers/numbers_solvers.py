import numpy as np
from .solver import Solver
from utils import number_to_base, is_prime, primes_lt, Symbols as S


class BaseConversionSolver(Solver):

    def play(self, problem):
        # TODO: only works with b2=10!!
        n = problem['n']
        b1 = problem['b1']
        b2 = problem['b2']

        num_in_b1 = number_to_base(n, b1)
        num_in_b2 = number_to_base(n, b2)

        self.print_symbols_ltr(
            num_in_b1, attention=True, mark_pos='num',
            orientation=1
        )
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

        num_in_by_rev = num_in_b1.copy()
        num_in_by_rev.reverse()

        for i, digit in enumerate(num_in_by_rev):
            self.mark_current_pos('s')
            # TODO why 3?
            self.move_left(3)
            self.print_symbol(S.add)
            self.print_number(digit, orientation=1)
            self.print_symbol(S.product, orientation=1)
            self.print_symbols_ltr(
                number_to_base(b1**i, b=b2), orientation=1)
            self.print_symbol(S.eq, orientation=1)
            self.print_number(b1**i*digit, orientation=1)
            self.make_step()
            self.go_to_mark('s')
            self.move_down()

        self.print_symbols_ltr(num_in_b2, attention=True,
                               reset=True)
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
