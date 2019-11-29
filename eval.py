import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds
import sys

from utils import Symbols
import params

args = params.getArgs()
print(args)

dataset = generators.sup_dataset_from_tfrecords([os.path.join(args.dataset_path, args.dataset) + args.split + '.tfrecords'])

model = tf.keras.models.load_model(args.model_path)

# Check its architecture
model.summary()


def is_finished(state):
  return np.any(state == Symbols.end)

for i, x in range(dataset.take(args.eval_size).enumerate()):

  state = x
  finished = False
  while not finished:
    res = model.predict(start_state)
    state += res
    finished = is_finished(state)



