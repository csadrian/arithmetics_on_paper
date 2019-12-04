import matplotlib
matplotlib.use('Agg')

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
import paper
import display

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

np.set_printoptions(threshold=sys.maxsize)

points = 0
total_points = 0
i = 0
for xy in preprocess(dataset).take(args.eval_size):
  i+=1
  x = xy[0][:]
  y = xy[1][:]
  pred_y = model.predict(x)
  pred_ys = np.argmax(pred_y, axis=-1)
  ys = np.argmax(y, axis=-1)
  xs = np.argmax(x, axis=-1)

  print(pred_ys.shape)
  print(ys.shape)

  bad = np.where(False == np.all(pred_ys == ys, axis=(1,2)))
  print(bad)
  badi = bad[0][0]
  print("xs", xs[badi].shape)
  print("ys", ys[badi].shape)
  print("pys", pred_ys[badi].shape)
  
  step1 = paper.Step(paper=xs[badi], attention=None)
  step2 = paper.Step(paper=pred_ys[badi], attention=None)
  step3 = paper.Step(paper=ys[badi], attention=None)
  display.plot_steps([step1, step2, step3], title="bad"+str(i), savefig=True)
  
  points += np.sum(np.all(pred_ys == ys, axis=(1,2)))
  total_points += args.batch_size
  print("points: ", points, ", total: ", total_points, ", precent: ", points/total_points)

quit()


for x in preprocess(dataset).take(args.eval_size):

  state = x[0][0]
  state = np.expand_dims(state, axis=0)
  state_s = np.argmax(state, axis=-1)
  print("state_s.shape:", state_s.shape)
  finished = False
  steps = 0

  while not finished and steps < 100:

    pred = model.predict(state)
    pred_s = np.argmax(pred, axis=-1)
    finished = is_finished(pred_s)
    state_s = state_s + pred_s
    print("state_s.shape:", state_s.shape)

    state = tf.one_hot(state_s, Symbols.NUM_SYMBOLS, axis=-1)
    print(state_s)


    steps += 1
    print(steps)
  print("Finished in " + str(steps))


