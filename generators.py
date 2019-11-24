import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.layers as layers
import problems
import solvers_milan as solvers
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

GRID_SIZE=15
def generator(Solver_, Problem_, N=10):
    problem_generator = Problem_().generator()
    solver = Solver_(GRID_SIZE)
    iterator = iter(solver.generator(problem_generator))
    solutions = []
    for i in range(N):
        solutions.append(next(iterator))
    return solutions


def generate_dataset_test():

    curriculum = [
        {'problem': problems.IsPrimeProblem, 'solver': solvers.IsPrimeSolverEasy, 'n': 100 },
        {'problem': problems.IsPrimeProblem, 'solver': solvers.IsPrimeSolverEasy, 'n': 100 }
    ]
    for curr in curriculum:
        # Train, validation and test is not completely separated now,
        # as a given problem can occur randomly in all sets.

        solutions = generator(Solver_=curr['solver'], Problem_=curr['problem'], N=curr['n'])
        records = solutions_to_pairs(solutions)
        write_tfrecords(records, 'test1.train')

        solutions = generator(Solver_=curr['solver'], Problem_=curr['problem'], N=curr['n'])
        records = solutions_to_pairs(solutions)
        write_tfrecords(records, 'test1.val')

        solutions = generator(Solver_=curr['solver'], Problem_=curr['problem'], N=curr['n'])
        records = solutions_to_pairs(solutions)
        write_tfrecords(records, 'test1.test')


def solutions_to_pairs(solutions):
    records = []
    sol_no = 0
    for steps in solutions:
        i = 0
        for first, second in zip(steps, steps[1:]):
            record_d = {}
            record_d['w'] = first['paper'].shape[0]
            record_d['h'] = first['paper'].shape[1]
            record_d['paper'] = first['paper'].astype(int)
            record_d['attention'] = first['attention'].astype(int)
            record_d['target'] = second['paper'] - first['paper']
            record_d['step'] = i
            record_d['sol_no'] = sol_no
            records.append(record_d)
            i += 1
        sol_no += 1
    return records


def write_tfrecords(records, name):
    file_path_prefix = ''
    result_tf_file = file_path_prefix + name + ".tfrecords"
    writer = tf.io.TFRecordWriter(result_tf_file)
    for record in records:
      feature_kvps = {
          'w': tf.train.Feature(int64_list=tf.train.Int64List(value=[record['w']])),
          'h': tf.train.Feature(int64_list=tf.train.Int64List(value=[record['h']])),
          'step': tf.train.Feature(int64_list=tf.train.Int64List(value=[record['step']])),
          'sol_no': tf.train.Feature(int64_list=tf.train.Int64List(value=[record['sol_no']])),
      }
      for key in ['paper', 'attention', 'target']:
          feature_kvps[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=record[key].ravel()))

      features = tf.train.Features(feature=feature_kvps)
      example = tf.train.Example(features=features)
      serialized = example.SerializeToString()
      writer.write(serialized)

def read_tfrecord(serialized_example):
    feature_description = {
        'w': tf.io.FixedLenFeature([], tf.int64),
        'h': tf.io.FixedLenFeature([], tf.int64),
        'paper': tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.int64, allow_missing=True),
        'attention': tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.int64, allow_missing=True),
        'target': tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.int64, allow_missing=True),
        'step': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    return example

def dataset_from_tfrecords(file_paths):
    tfrecord_dataset = tf.data.TFRecordDataset(file_paths)
    parsed_dataset = tfrecord_dataset.map(read_tfrecord)
    return parsed_dataset

def sup_dataset_from_tfrecords(file_paths):
    dataset = dataset_from_tfrecords(file_paths)
    return dataset.map(lambda r: (tf.reshape(r['paper'], (r['w'], r['h'])), tf.reshape(r['target'], (r['w'], r['h']))))

if __name__ == "__main__":
    generate_dataset_test()