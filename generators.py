import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds

SIGN_ADD = 11
GRID_SIZE = 10

def number_to_base(n, b=10):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


def print_number(paper, n, x, y):
    n_in_base = number_to_base(n)
    for i in range(len(n_in_base)):
        paper[x][y - i] = n_in_base[-(i+1)]
    return x, y-len(n_in_base)

def make_addition(a, b, grid_size=10):
    steps = []

    paper = -1*np.ones(shape=(grid_size, grid_size))
    start = (random.randint(4,6), random.randint(4,6))

    digits_a = number_to_base(a)
    digits_b = number_to_base(b)
    digits_num_max = max(len(digits_a), len(digits_b))

    x = start[0]
    y = start[1]

    x, y = print_number(paper, a, x, y)
    y = start[1]
    x += 1
    x, y = print_number(paper, b, x, y)
    paper[x, y] = SIGN_ADD

    steps.append(paper.copy())

    y = start[1]
    x += 1
    carry = 0
    for i in range(1,digits_num_max+1):
        curr_sum = digits_a[-i] + digits_b[-i] + carry
        carry = curr_sum // 10 if curr_sum >= 10 else 0
        x, y = print_number(paper, curr_sum % 10, x, y)
        steps.append(paper.copy())
    return steps

def generate_dataset(N=1000, grid_size=10):
    xs, ys = [], []
    for i in range(N):
        a = random.randint(100, 500)
        b = random.randint(100, 500)
        steps = make_addition(a, b, grid_size=grid_size)
        for first, second in zip(steps, steps[1:]):
            xs.append(first)
            ys.append(second)

    return np.array(xs), np.array(ys)

