import numpy as np
import random
# import tensorflow as tf
# import tensorflow.keras.layers as layers

class Problem:
    order = NotImplemented

    def generator(self):
        while True:
            yield self.generate_one()

    generate_one = NotImplemented

class BinaryOperationProblem(Problem):

    def __init__(self, lim_a, lim_b):
        self.lim_a = lim_a
        self.lim_b = lim_b

    def generate_one(self):
        return {'a': np.random.randint(self.lim_a),
                'b': np.random.randint(self.lim_b)}

class AdditionProblem(BinaryOperationProblem):
    pass

class EasyAdditionProblem(AdditionProblem):
    order = 10
    def __init__(self):
        super().__init__(10, 10)

class EasyProductProblem(AdditionProblem):
    order = 20
    def __init__(self):
        super().__init__(10, 10)

class HardAdditionProblem(AdditionProblem):
    order = 100
    def __init__(self):
        super().__init__(100, 100)


def _problem_example():
    problem = HardAdditionProblem()
    iterator = HardAdditionProblem.generate()
    for i in range(10):
        print(next(iterator))
