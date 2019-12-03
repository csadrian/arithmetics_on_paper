import numpy as np
import decimal #, sympy
from utils import number_to_base, Symbols as S


class Step:

    def __init__(self, paper, attention, solver=None):
        self.paper = paper
        self.attention = attention
        self.solver = solver

    def __getitem__(self, key):
        return getattr(self, key)

def print_func(func):
    def inner(*args, **kwargs):
        self = args[0]
        x, y = self._x, self._y
        reserve_pos = kwargs.pop('preserve_pos', False)
        res = func(*args, **kwargs)
        if reserve_pos:
            self._set_position(x, y)
        return res
    return inner

def reset_arg(func):
    def inner(*args, **kwargs):
        self = args[0]
        if kwargs.pop('reset', False):
            self.reset_attention()
        res = func(*args, **kwargs)
        return res
    return inner


class PaperWithNumbers:

    def __init__(self, grid_size, startx=3, starty=8):
        self.shape = (grid_size, grid_size)
        self.paper = np.zeros(shape=self.shape, dtype=np.int32)

        self.reset_attention()
        self.steps = []
        self._marked_cells = dict()
        self._marked_ranges = dict()

        self._mark_cell('question', (0, 0))
        self.go_to_mark('question')
        self.print_symbol(S.question)
        self._mark_cell('answer', (1, 1))
        self.go_to_mark('answer')
        self.move_left()
        self.print_symbol(S.answer, orientation=-1)

        self._x = startx
        self._y = starty
        self.mark_current_pos('start')

    def reset_attention(self):
        self.attention = np.zeros(shape=self.shape)

    def make_step(self, solver=None):
        self.steps.append(Step(self.paper.copy(),
                               self.attention.copy(),
                               solver))

    def get_steps(self):
        return self.steps

    def _mark_cell(self, name, pos):
        self._marked_cells[name] = pos

    def _mark_range(self, name, range):
        # range: list of x,y points.
        # sorted ltr!
        self._marked_ranges[name] = sorted(range)

    def _set_position(self, x, y):
        self._x, self._y = x, y

    def go_to_mark(self, name):
        mark = self._marked_cells[name]
        self._x, self._y = mark

    def go_to_mark_range(self, name, end=False):
        """
        end : bool
            if True: go to end of range, otherwise beginning
        """
        mark = self._marked_ranges[name]
        self._x, self._y = mark[-1] if end else mark[0]

    def _value_at_position(self, x, y):
        return self.paper[x, y]

    def value_at_position(self):
        return self._value_at_position(self._x, self._y)

    def get_marks_at_pos(self):
        x, y = self._x, self._y
        res = []
        for mark in self._marked_cells:
            if self._marked_cells[mark] == (x, y):
                res += mark
        return res

    def _check_cell_emptiness(self, pos):
        if self.paper[pos[0], pos[1]] != 0:
            print("Warning: content of a non-empty grid cell is going to be overwritten on the paper!")

    @reset_arg
    def set_attention_mark(self, name):
        self.set_attention([self._marked_cells[name]])

    def remove_attention_mark(self, name):
        self.remove_attention([self._marked_cells[name]])

    @reset_arg
    def set_attention_mark_range(self, name):
        self.set_attention(self._marked_ranges[name])

    def remove_attention_mark_range(self, name):
        self.remove_attention(self._marked_ranges[name])

    def mark_current_pos(self, name, vertical_offset=0, horizontal_offset=0):
        self._mark_cell(name,
                        (self._x + vertical_offset, self._y + horizontal_offset))

    @reset_arg
    def set_attention_current_pos(self):
        self.set_attention([(self._x, self._y)])

    @print_func
    @reset_arg
    def print_symbols_ltr(self, ns, attention=False,
                          orientation=1, mark_pos=False,
                          step_by_step=False):
        """
        mark_pos : bool or str
            False/0 if no marking needed, name of the mark otherwise
        """
        x, y = self._x, self._y
        ns = list(ns)
        if orientation < 0:
            ns.reverse()
        for i, symbol in enumerate(ns):
            self._check_cell_emptiness((x, y))
            self.paper[x, y] = symbol
            if attention:
                self.attention[x, y] = 1
            if step_by_step:
                self.make_step()
            y += orientation
        if mark_pos:
            range_ = []
            for i in range(len(ns)):
                range_.append((x, y+orientation*i))
            self._mark_range(mark_pos, range_)
        self._set_position(x, y)

    @print_func
    @reset_arg
    def print_symbol(self, n, step_by_step=False, attention=False,
                     orientation=1, mark_pos=False):
        x, y = self._x, self._y
        self._check_cell_emptiness((x, y))
        self.paper[x, y] = n
        if attention:
            self.attention[x, y] = 1
        if step_by_step:
            self.make_step()
        if mark_pos:
            self._mark_cell(mark_pos, (self._x, self._y))
        self._set_position(x, y + orientation)

    @print_func
    @reset_arg
    def print_number(self, n, step_by_step=False, attention=False, 
                      orientation=-1, mark_pos=False, solver=None):
        x, y = self._x, self._y
        if n < 0:
            self.print_symbol(S.sub, orientation=1, step_by_step=step_by_step)
            n = n.__mul__(-1)
        if isinstance(n, decimal.Decimal) and len(str(n).split('.')) == 2:
            integer_str, fractional_str = [item for item in str(n).split('.')]
            if orientation < 0:
                part1, part2 = fractional_str[::-1], integer_str[::-1]
            else:
                part1, part2 = integer_str, fractional_str
            for letter in part1:
                self.print_symbol(int(letter) + 1, step_by_step, attention, orientation, mark_pos)
            self.print_symbol(S.decimal, step_by_step, attention, orientation, mark_pos)
            for letter in part2:
                self.print_symbol(int(letter) + 1, step_by_step, attention, orientation, mark_pos)
            return
        elif isinstance(n, decimal.Decimal) and str(n).isdigit():
            integer_str = str(n)
            for letter in integer_str:
                self.print_symbol(int(letter) + 1, step_by_step, attention, orientation, mark_pos)
            return

        n_in_base = number_to_base(n)
        if orientation > 0:
            n_in_base.reverse()
        for i in range(len(n_in_base)):
            offset = i * orientation
            self._check_cell_emptiness((x, y + offset))
            self.paper[x, y + offset] = n_in_base[-(i+1)] + 1
            if attention:
                self.attention[x, y + offset] = 1
            if step_by_step:
                self.make_step(solver=solver)
        if mark_pos:
            res = []
            for i in range(len(n_in_base)):
                res.append((x, y+orientation*i))
            self._mark_range(mark_pos, res)
        self._set_position(x, y+orientation*len(n_in_base))

    @reset_arg
    def set_attention(self, points):
        for (x, y) in points:
            self.attention[x, y] = 1

    def remove_attention(self, points):
        for (x, y) in points:
            self.attention[x, y] = 0

    def move_right(self, n=1):
        self._y = self._y + n

    def move_down(self, n=1):
        self._x = self._x + n 

    def move_left(self, n=1):
        return self.move_right(-1*n)

    def move_up(self, n=1):
        return self.move_down(-1*n)

    def _word_at_position(self, orientation=-1):
        # word: sequence of symbols without empty sign
        self.mark_current_pos('gcn')
        res = []
        while self.value_at_position() in range(1, 11):
            res.append((self._x, self._y))
            self.move_right(orientation)
        self.go_to_mark('gcn')
        if orientation < 0:
            res = reversed(res)
        return res

    def set_attention_word(self, orientation=-1):
        # at the last digit of the number
        self.set_attention(self._word_at_position(orientation))

    def _get_word_at_position(self, orientation=-1):
        res = []
        for cell in self._word_at_position(orientation):
            res.append(self._value_at_position(*cell))
        return ''.join((str(digit-1) for digit in res))

    def get_number_at_position(self, orientation=-1):
        word = self._get_word_at_position(orientation)
        if word == '':
            return 0
        return int(self._get_word_at_position(orientation))
