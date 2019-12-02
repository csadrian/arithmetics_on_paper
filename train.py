import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds
import sys
import os

import params
import generators
from utils import Symbols
import paper
import display

from attn_augconv import AttentionAugmentation2D, augmented_conv2d

args = params.getArgs()
print(args)

# set random seed
#np.random.seed(10)

tf.compat.v1.enable_eager_execution()

NUM_SYMBOLS = 33
GRID_SIZE = 22

dataset_prefix = os.path.join(args.dataset_path, args.dataset)

xy_train =  generators.sup_dataset_from_tfrecords([dataset_prefix+'.train.tfrecords'])
xy_val   =  generators.sup_dataset_from_tfrecords([dataset_prefix+'.interpolate.tfrecords'])
xy_test  =  generators.sup_dataset_from_tfrecords([dataset_prefix+'.extrapolate.tfrecords'])


# Instantiate a simple classification model
inp = tf.keras.layers.Input(shape=(GRID_SIZE, GRID_SIZE, NUM_SYMBOLS))

out = layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu)(inp)


out = augmented_conv2d(out, filters=200, kernel_size=(3, 3),
                         depth_k=0.2, depth_v=0.2,  # dk/v (0.2) * f_out (20) = 4
                         num_heads=4, relative_encodings=True)
out = layers.Activation('relu')(out)

out = layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu)(out)
out = layers.BatchNormalization()(out)

out = augmented_conv2d(out, filters=200, kernel_size=(3, 3),
                         depth_k=0.2, depth_v=0.2,  # dk/v (0.2) * f_out (20) = 4
                         num_heads=4, relative_encodings=True)
out = layers.Activation('relu')(out)


out = augmented_conv2d(out, filters=200, kernel_size=(3, 3),
                         depth_k=0.2, depth_v=0.2,  # dk/v (0.2) * f_out (20) = 4
                         num_heads=4, relative_encodings=True)
out = layers.Activation('relu')(out)

#out = layers.BatchNormalization()(out)

#out = layers.Conv2D(256, (5,5), padding='same', activation=tf.nn.relu)(out)
#out = layers.Conv2D(2 * depth_k + depth_v, (5,5), padding='same', activation=tf.nn.relu)(out)
#out = AttentionAugmentation2D(depth_k, depth_v, num_heads)(out)


#out = layers.Conv2D(256, (3,3), padding='same', activation=tf.nn.relu)(out)
#out = layers.Conv2D(256, (3,3), padding='same', activation=tf.nn.relu)(out)
#out = layers.Conv2D(256, (3,3), padding='same', activation=tf.nn.relu)(out)
out = layers.Conv2D(NUM_SYMBOLS, (3,3), padding='same')(out)

model = tf.keras.Model(inputs=inp, outputs=out)

# Instantiate a logistic loss function that expects integer targets.

"""
eps = 1e-8
def my_loss(y_true, y_pred):
    ans = -(y_true*tf.log(y_pred + eps) + (1-y_true)*tf.log(y_pred + eps))
    return ans
def xxmy_loss(y_true, y_pred):
    #ans =  tf.square(y_true-y_pred) * tf.boolean_mask(y_true, tf.equal(y_true, 0)) * 0.0001 
    #ans += tf.square(y_true-y_pred) * tf.boolean_mask(y_true, tf.not_equal(y_true, 0)) * 0.0001 
    ans =  tf.square(tf.softmax(y_true-y_pred) * (y_true != 0) * 1.0 
    ans += tf.square(y_true-y_pred) * (y_true == 0) * 0.00000     
    return ans
"""    

cc_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.0)

def weighted_loss(y_true, y_pred):
    weights = tf.ones([50, GRID_SIZE, GRID_SIZE]) * 0.01 + tf.clip_by_value(tf.reduce_sum(y_true, axis=-1), 0.0, 1.0)
    return cc_loss(y_true, y_pred, weights)
    #a = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred, axis=-1)
    #return tf.reduce_sum(a * weights)


# Instantiate an accuracy metric.
accuracy = tf.keras.metrics.CategoricalAccuracy()

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer,
              loss=weighted_loss,
              metrics=[accuracy])

# Instantiate some callbacks
callbacks = []
if args.model_path is not None:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=args.model_path, save_best_only=True))

def preprocess(ds):
    ds = ds.batch(50, drop_remainder=True).prefetch(100)
    ds = ds.shuffle(10000)
    ds = ds.map(lambda x, y: (tf.one_hot(x, NUM_SYMBOLS, axis=-1), tf.one_hot(y, NUM_SYMBOLS, axis=-1)))
    return ds

def preprocess_test(ds):
    ds = ds.shuffle(10000)
    ds = ds.map(lambda x, y: tf.one_hot(x, NUM_SYMBOLS, axis=-1))
    return ds

train_dataset = preprocess(xy_train).repeat()
val_dataset = preprocess(xy_val).repeat()
test_dataset = preprocess_test(xy_val)


model.fit(train_dataset,
          validation_data=val_dataset,
          validation_steps=100,
          epochs=25,
          steps_per_epoch=1000,
          callbacks=callbacks)

if args.model_path is not None:
    model.save(args.model_path)

np.set_printoptions(threshold=sys.maxsize)

x_test = [e.numpy() for e in test_dataset]
x_test = np.array(x_test)

preds = model.predict(x_test)

for i in range(10):
    paper_x = np.argmax(x_test[i], axis=-1)
    paper_pred = np.argmax(preds[i], axis=-1)
    step1 = paper.Step(paper=paper_x, attention=None)
    step1.paper = paper_x
    step2 = paper.Step(paper=paper_pred, attention=None)
    step2.paper = paper_pred
    display.plot_steps([step1, step2], title=str(i))



"""
preds = model.predict(test_large_dataset)

for i in range(10):
    print(x_test_large[i])
    print(np.argmax(preds[i], axis=-1))
"""