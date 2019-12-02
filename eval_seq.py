import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds
import sys
import os

from attn_augconv import AttentionAugmentation2D, augmented_conv2d

from utils import Symbols
import params
import generators
#from tensorflow.keras.utils.generic_utils import get_custom_objects

args = params.getArgs()
print(args)


tf.compat.v1.enable_eager_execution()

dataset = generators.sup_dataset_from_tfrecords([os.path.join(args.dataset_path, args.dataset) + '.' + args.split + '.tfrecords'])

model = tf.keras.models.load_model(os.path.join(args.model_path, args.model_name), custom_objects={'AttentionAugmentation2D': AttentionAugmentation2D, 'weighted_loss': tf.keras.losses.mse})

def preprocess(ds):
    ds = ds.batch(args.batch_size, drop_remainder=True).prefetch(100)
    ds = ds.shuffle(10000)
    ds = ds.map(lambda x, y: (tf.one_hot(x, Symbols.NUM_SYMBOLS, axis=-1), tf.one_hot(y, Symbols.NUM_SYMBOLS, axis=-1)))
    return ds

def preprocess_test(ds):
    ds = ds.shuffle(10000)
    ds = ds.map(lambda x, y: tf.one_hot(x, Symbols.NUM_SYMBOLS, axis=-1))
    return ds



# Check its architecture
model.summary()

def is_finished(state):
  return np.any(state == Symbols.end)

for i, x in preprocess(dataset).take(args.eval_size).enumerate():

  state = x[0]
  finished = False
  steps = 0
  while not finished and steps < 100:
    res = model.predict(state)
    res = np.argmax(res, axis=-1)
    res = tf.one_hot(res, Symbols.NUM_SYMBOLS, axis=-1)
    state += res
    finished = is_finished(state)
    steps += 1
    print(steps)

  print("Finished")


