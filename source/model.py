from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# Build the model with 9 hidden(excluding input and output layer)


def build_model(num_pca_x, num_input):
    num_hidden = int(1.5*num_pca_x)
    model = keras.Sequential([
        layers.Dense(num_hidden, activation=tf.nn.relu, input_shape=[num_input]),
        layers.Dense(num_hidden, activation=tf.nn.relu),
        layers.Dense(num_hidden, activation=tf.nn.relu),
        layers.Dense(num_hidden, activation=tf.nn.relu),
        layers.Dense(num_hidden, activation=tf.nn.relu),
        layers.Dense(num_hidden, activation=tf.nn.relu),
        layers.Dense(num_hidden, activation=tf.nn.relu),
        layers.Dense(num_hidden, activation=tf.nn.relu),
        layers.Dense(num_hidden, activation=tf.nn.relu),
        layers.Dense(num_pca_x)
    ])
    return model







