import numpy as np
from sympy import *
from .problem import *

class TwoPolynomialProblem(Problem):

    def __init__(self, lim, deg=None):
        if deg is None:
            self.lim = lim
        else:
            self.lim = [lim[0] for _ in range(deg)]

    def generate_one(self):

        x = symbols('x')

        return {'a' : sum([(x ** i) * np.random.randint(self.lim[i][0], self.lim[i][1]) for i in range(len(self.lim))]),
                'b' : sum([(x ** i) * np.random.randint(self.lim[i][0], self.lim[i][1]) for i in range(len(self.lim))])}

class SinglePolynomialProblem(Problem):

    def __init__(self, lim, mult=False):
        self.mult = mult
        self.lim = lim

    def generate_one(self):

        x = symbols('x')

        pass

class AddPolynomialProblem(TwoPolynomialProblem):
    pass

class ComposeProblem(TwoPolynomialProblem):
    pass

class CollectProblem(SinglePolynomialProblem):

    def __init__(self, lim):
        super().__init__(lim, mult=True)

class EvaluateProblem(SinglePolynomialProblem):
    pass
