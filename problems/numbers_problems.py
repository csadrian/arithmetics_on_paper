import numpy as np
from .problem import *


class BaseConversionProblem(Problem):
    def __init__(self, lim, b1, b2):
        self.lim = lim
        self.b1 = b1
        self.b2 = b2

    def generate_one(self):
        n = np.random.randint(self.lim)
        return {'n': n,
                'b1': self.b1,
                'b2': self.b2}


class BaseConversion2to3(BaseConversionProblem):
    order = 10

    def __init__(self):
        super().__init__(100, 2, 3)


class IsPrimeProblem(OneNumProblem):
    order = 10

    def __init__(self):
        super().__init__(30)


class IsPrimeProblemHard(Problem):
    order = 10

    def __init__(self, lim_a):
        self.lim_a = lim_a

    def generate_one(self):
        return {'a': np.random.randint(self.lim_a)}


class PlaceValueProblem(Problem):
    #for a given lim the maximal number generated is 10^{lim}
    def __init__(self, lim):
        self.lim = lim

    def generate_one(self):
        return {'number' : np.random.randint(10 **(self.lim)),
                'place' : np.random.randint(1, self.lim)}
