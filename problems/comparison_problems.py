import numpy as np
from .problems import *


class PaircomparisonProblem(BinaryOperationProblem):
    order = 10
    def __init__(self):
        super().__init__(100, 100)


class SortProblem(Problem):

    def __init__(self, lim, size=5):
        self.lim = lim
        self.size = size

    def generate_one(self):
        return {'ns': np.random.randint(1, self.lim, size=self.size)}
