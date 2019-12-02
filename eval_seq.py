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


dataset = generators.sup_dataset_from_tfrecords([os.path.join(args.dataset_path, args.dataset) + args.split + '.tfrecords'])

#get_custom_objects().update({"weighted_loss": tensorflow.keras.losses.mse})
model = tf.keras.models.load_model(os.path.join(args.model_path, args.model_name), custom_objects={'AttentionAugmentation2D': AttentionAugmentation2D, 'weighted_loss': tf.keras.losses.mse})
#model = tf.keras.models.load_model(os.path.join(args.model_path, args.model_name))

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
    print(res)
  print("Finished")


