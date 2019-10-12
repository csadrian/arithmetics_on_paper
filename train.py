import numpy as np
import random
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds

import generators

NUM_SYMBOLS = 18
GRID_SIZE = 12

x_train, y_train = generators.generate_dataset(N=20000, grid_size=GRID_SIZE)
x_val  , y_val   = generators.generate_dataset(N=10000, grid_size=GRID_SIZE)
x_test , y_test  = generators.generate_dataset(N=10000, grid_size=GRID_SIZE)

x_test_large, y_test_large = generators.generate_dataset_addition(N=10000, grid_size=GRID_SIZE, size=5)



# Instantiate a simple classification model
inp = tf.keras.layers.Input(shape=(GRID_SIZE, GRID_SIZE, NUM_SYMBOLS))

out = layers.Conv2D(256, (3,3), padding='same', activation=tf.nn.relu)(inp)
out = layers.Conv2D(256, (3,3), padding='same', activation=tf.nn.relu)(out)
out = layers.Conv2D(256, (3,3), padding='same', activation=tf.nn.relu)(out)
out = layers.Conv2D(256, (3,3), padding='same', activation=tf.nn.relu)(out)
out = layers.Conv2D(256, (3,3), padding='same', activation=tf.nn.relu)(out)
out = layers.Conv2D(NUM_SYMBOLS, (3,3), padding='same')(out)

"""
out = layers.Flatten()(inp)
out = layers.Dense(1256, activation=tf.nn.relu)(out)
out = layers.Dense(1256, activation=tf.nn.relu)(out)
out = layers.Dense(1256, activation=tf.nn.relu)(out)
out = layers.Dense(1256, activation=tf.nn.relu)(out)
out = layers.Dense(GRID_SIZE*GRID_SIZE*NUM_SYMBOLS, activation=tf.nn.relu)(out)
out = layers.Reshape((GRID_SIZE,GRID_SIZE,NUM_SYMBOLS))(out)
"""
model = tf.keras.Model(inputs=inp, outputs=out)

# Instantiate a logistic loss function that expects integer targets.
loss = tf.keras.losses.MeanSquaredError()

# Instantiate an accuracy metric.
accuracy = tf.keras.metrics.CategoricalAccuracy()

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=[accuracy])

# Instantiate some callbacks
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='my_model.keras',
                                                save_best_only=True)]

def preprocess(ds):
    ds = ds.batch(50)
    ds = ds.map(lambda x, y: (tf.one_hot(x, NUM_SYMBOLS, axis=-1), tf.one_hot(y, NUM_SYMBOLS, axis=-1)))
    return ds

def preprocess_test(ds):
    ds = ds.batch(50)
    ds = ds.map(lambda x: tf.one_hot(x, NUM_SYMBOLS, axis=-1))
    return ds

train_dataset = preprocess(tf.data.Dataset.from_tensor_slices((x_train, y_train))).repeat()
val_dataset = preprocess(tf.data.Dataset.from_tensor_slices((x_val, y_val)))
test_dataset = preprocess_test(tf.data.Dataset.from_tensor_slices(x_test))
test_large_dataset = preprocess_test(tf.data.Dataset.from_tensor_slices(x_test_large))

model.fit(train_dataset,
          validation_data=val_dataset, 
          epochs=10,
          steps_per_epoch=1000,
          callbacks=callbacks)

preds = model.predict(test_dataset)

for i in range(10):
    print(x_test[i])
    print(np.argmax(preds[i], axis=-1))

preds = model.predict(test_large_dataset)

for i in range(10):
    print(x_test_large[i])
    print(np.argmax(preds[i], axis=-1))
