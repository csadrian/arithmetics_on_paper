import numpy as np
import random
from common import number_to_base
# import tensorflow as tf
# import tensorflow.keras.layers as layers


# ABSTRACT CLASSES
class Problem:

    def generator(self):
        while True:
            yield self.generate_one()

    def generate_one(self):
        raise NotImplementedError


class BinaryOperationProblem(Problem):

    def __init__(self, lim_a, lim_b):
        self.lim_a = lim_a
        self.lim_b = lim_b

    def generate_one(self):
        return {'a': np.random.randint(self.lim_a),
                'b': np.random.randint(self.lim_b)}


class OneNumProblem(Problem):

    def __init__(self, lim_a):
        self.lim_a = lim_a

    def generate_one(self):
        return {'a': np.random.randint(self.lim_a)}


class AdditionProblem(BinaryOperationProblem):
    pass

class MultiplicationProblem(BinaryOperationProblem):
    pass


class BaseConversionProblem(Problem):
    def __init__(self, lim, b1, b2):
        self.lim = lim
        self.b1 = b1
        self.b2 = b2

    def generate_one(self):
        n = np.random.randint(self.lim)
        return {
            'n': n,
            'b1': self.b1,
            'b2': self.b2
        }

class AddSubMultipleProblem(Problem):
    #lim is a list of [lower_lim, upper_lim] pairs
    def __init__(self, lim):
        self.lim = lim
    
    def generate_one(self):
        return {
                'coefs' : [np.random.randint(self.lim[i][0], self.lim[i][1]) for i in range(len(self.lim))],
                'ops' : [np.random.randint(11, 12) for i in range(len(self.lim) - 1)]        
                }

class PlaceValueProblem(Problem):
    #for a given lim the maximal number generated is 10^{lim}
    def __init__(self, lim):
        self.lim = lim

    def generate_one(self):
        return {'number' : np.random.randint(10 **(self.lim)),
                'place' : np.random.randint(1, self.lim)}

# NON-ABSTRACT CLASSES
class BaseConversion2to3(BaseConversionProblem):
    order = 10
    
    def __init__(self):
        super().__init__(100, 2, 3)

class PaircomparisonProblem(BinaryOperationProblem):
    order = 10
    def __init__(self):
        super().__init__(100, 100)

class EasyAdditionProblem(AdditionProblem):
    order = 10
    def __init__(self):
        super().__init__(10, 10)


class EasyMultiplicationProblem(MultiplicationProblem):
    order = 20
    def __init__(self):
        super().__init__(10, 10)

class HardMultiplicationProblem(MultiplicationProblem):
    order = 200
    def __init__(self):
        super().__init__(1000, 1000)


class HardAdditionProblem(AdditionProblem):
    order = 100
    def __init__(self):
        super().__init__(100, 100)

class DivisionProblem(BinaryOperationProblem):
    order = 20
    def __init__(self):
        super().__init__(1000, 1000)

class IsPrimeProblem(OneNumProblem):
    order = 10

    def __init__(self):
        super().__init__(30)


class IsPrimeProblemHard(Problem):
    # 50% prime, 50% not.
    order = 10

    def __init__(self, lim_a):
        self.lim_a = lim_a

    def generate_one(self):
        return {'a': np.random.randint(self.lim_a)}

def _problem_example():
    problem = HardAdditionProblem()
    iterator = HardAdditionProblem.generate()
    for i in range(10):
        print(next(iterator))


__all__ = [
    'EasyAdditionProblem',
    'HardAdditionProblem',
    'EasyMultiplicationProblem',
    'HardMultiplicationProblem',
    'IsPrimeProblem',
    'DivisionProblem',
    'PaircomparisonProblem'
]
