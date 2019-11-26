import numpy as np
from solvers import Solver, AddSolver

from common import number_to_base, is_prime, primes_lt, Symbols as S


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

class RoundNumber(Solver):
    pass

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

class SortSolver(Solver):

    def longest_decimal(self, ns):
        lens = [len(number_to_base(n)) for n in ns]
        return np.max(lens)

    def get_number_at_position(self, orientation=-1):
        return int(self.get_word_at_position(orientation))

    def play(self, problem):
        ns = problem['ns']

        self.print_symbol(S.sort, orientation=0)

        for n in ns:
            self.move_down()
            self.print_number(n, orientation=-1, preserve_pos=True)

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
