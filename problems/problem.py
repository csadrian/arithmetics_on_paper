import numpy as np


class Problem:

    def generator(self):
        while True:
            yield self.generate_one()

    def generate_one(self):
        raise NotImplementedError


class BinaryOperationProblem(Problem):

    def __init__(self, params):
        #self.lim_a = lim_a
        #self.lim_b = lim_b
        self.params = params

    def generate_one(self):
        return {'a': np.random.randint(self.lim_a),
                'b': np.random.randint(self.lim_b)}


class OneNumProblem(Problem):

    def __init__(self, lim_a):
        self.lim_a = lim_a

    def generate_one(self):
        return {'a': np.random.randint(self.lim_a)}
