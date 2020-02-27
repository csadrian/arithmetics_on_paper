import numpy as np
from .problem import *


class AdditionProblem(BinaryOperationProblem):
    
    def __init__(self, params, base_lim=[10, 10]):
        super().__init__(params)
        self.base_lim = base_lim

    def generate_one(self):
        return super().generate_one.update({'base' : np.random.randint(self.base_lim[0], self.base_lim[1])})


class MultiplicationProblem(BinaryOperationProblem):
    pass


class AddSubMultipleProblem(Problem):
    #lim is a list of [lower_lim, upper_lim] pairs
    def __init__(self, lim):
        self.lim = lim

    def generate_one(self):
        return {'coefs' : [np.random.randint(self.lim[i][0], self.lim[i][1]) for i in range(len(self.lim))],
                'ops' : [np.random.randint(11, 12) for i in range(len(self.lim) - 1)]}

class AddOrSubProblem(BinaryOperationProblem):
    pass


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
    pass


__all__=['DivisionProblem']
