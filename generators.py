import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.layers as layers
# import tensorflow_datasets as tfds

def generate_with_generators(generators, N, grid_size):
    xs, ys = [], []

    for generator in generators:
        for i in range(N):
            paper = PaperWithNumbers(grid_size)
            generator.set_paper(paper)
            generator.generate()
            steps = generator.get_steps()
            for first, second in zip(steps, steps[1:]):
                xs.append(first)
                ys.append(second)

    xs = np.array(xs)
    ys = np.array(ys)

    p = np.random.permutation(len(xs))
    xs, ys = xs[p].astype(int), ys[p].astype(int)

    return xs, ys-xs


def generate_dataset(N=1000, grid_size=10):
    
    generators = [
        AddSolver(mits=(1000, 5000), b_limits=(1000, 5000)),
        AddSolver(a_limits=(100, 500), b_limits=(100, 500)),
        AddSolver(a_limits=(10, 50),   b_limits=(10, 50)),
        SubstractSolver(a_limits=(500, 900), b_limits=(100, 400)),
        IsPrimeSolver(limit=33),
        IsDivisibleBySolver(limit=20),
        IsDivisibleBySolver(limit=50),
        FactorizeSolver(limit=50),
    ]
    return generate_with_generators(generators, N=N, grid_size=grid_size)

def generate_dataset_addition(N=1000, grid_size=10, size=3):
    
    generators = [
        AddSolver(a_limits=(1*10**(size-1), 5*10**(size-1)), b_limits=(1*10**(size-1), 5*10**(size-1))),
    ]

    return generate_with_generators(generators, N=N, grid_size=grid_size)
