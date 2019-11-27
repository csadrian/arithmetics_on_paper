import numpy as np
from .problems import *


class AdditionProblem(BinaryOperationProblem):
    pass


class MultiplicationProblem(BinaryOperationProblem):
    pass


class AddSubMultipleProblem(Problem):
    #lim is a list of [lower_lim, upper_lim] pairs
    def __init__(self, lim):
        self.lim = lim

    def generate_one(self):
        return {'coefs' : [np.random.randint(self.lim[i][0], self.lim[i][1]) for i in range(len(self.lim))],
                'ops' : [np.random.randint(11, 12) for i in range(len(self.lim) - 1)]}


class EasyAdditionProblem(AdditionProblem):
    order = 10
    def __init__(self):
        super().__init__(10, 10)


class HardAdditionProblem(AdditionProblem):
    order = 100
    def __init__(self):
        super().__init__(100, 100)


class EasyMultiplicationProblem(MultiplicationProblem):
    order = 20
    def __init__(self):
        super().__init__(10, 10)


class HardMultiplicationProblem(MultiplicationProblem):
    order = 200
    def __init__(self):
        super().__init__(1000, 1000)


class DivisionProblem(BinaryOperationProblem):
    order = 20
    def __init__(self):
        super().__init__(1000, 1000)

__all__=['DivisionProblem']