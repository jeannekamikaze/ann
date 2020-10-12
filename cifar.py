#!/usr/bin/python3
#
# Chapter 2 of "Generative Deep Learning: Teaching Machines to Paint, Write,
# Compose, and Play".

# Get Tensorflow to shut up.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from keras.datasets import cifar10
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


NUM_CLASSES = 10


def normalize(pixels: np.array):
  """Map the given pixels array from integers in the range 0..255 to float32s
  in the range 0..1."""
  return pixels.astype('float32') / 255.0


def build_model(tag: str) -> Model:
  """Build a model for prediction.

  Args:
    tag: Either 'dense' or 'convolutional'.
  """
  if tag == "dense":
    model = Sequential([
      Flatten(input_shape=(32, 32, 3)),
      Dense(200, activation='relu'),
      Dense(150, activation='relu'),
      Dense(NUM_CLASSES, activation='softmax')
    ])
    # The same model built using the Functional API.
    # input_layer = Input(shape=(32, 32, 3))
    # x = Flatten()(input_layer)
    # x = Dense(units=200, activation = 'relu')(x)
    # x = Dense(units=150, activation = 'relu')(x)
    # output_layer = Dense(units=10, activation = 'softmax')(x)
    # model = Model(input_layer, output_layer)
  elif tag == "convolutional":
    model = Sequential([
      Input(shape=(32,32,3)),
      Conv2D(filters=32, kernel_size=3, strides=1, padding='same'),
      BatchNormalization(),
      LeakyReLU(),
      Conv2D(filters=32, kernel_size=3, strides=2, padding='same'),
      BatchNormalization(),
      LeakyReLU(),
      Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
      BatchNormalization(),
      LeakyReLU(),
      Conv2D(filters=64, kernel_size=3, strides=2, padding='same'),
      BatchNormalization(),
      LeakyReLU(),
      Flatten(),
      Dense(128),
      BatchNormalization(),
      LeakyReLU(),
      Dropout(rate=0.5),
      Dense(NUM_CLASSES, activation='softmax')
    ])
  else:
    raise ValueError("tag must be 'dense' or 'convolutional'")
  model.build()
  return model


def train_model(model: Model, x_train: np.array, y_train: np.array):
  """Train the model."""
  optimizer = Adam(lr=0.0005)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size = 12, epochs = 10, shuffle = True)


def plot_results(model: Model, x_test, y_test):
  """Plot a random subset of predictions on the test data."""

  def prob_to_class(probabilities: np.array):
    """Run predictions on a set of images."""
    assert probabilities.shape[1] == 10 # Should have 10 probabilities.
    CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                        'frog', 'horse', 'ship', 'truck'])
    predictions = CLASSES[np.argmax(probabilities, axis=-1)]
    return predictions

  predicted = prob_to_class(model.predict(x_test))
  actual = prob_to_class(y_test)

  total_images = 10 # Number of images to show.
  image_indices = np.random.choice(range(len(x_test)), total_images)

  fig = plt.figure(figsize=(15, 3))
  fig.subplots_adjust(hspace=0.4, wspace=0.4)

  for i, image_index in enumerate(image_indices):
    img = x_test[image_index]
    ax = fig.add_subplot(1, total_images, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred = ' + str(predicted[image_index]), fontsize=10,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, 'act = ' + str(actual[image_index]), fontsize=10,
            ha='center', transform=ax.transAxes)
    ax.imshow(img)

  plt.show()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("arch", type=str, choices=["dense", "convolutional"])
  args = parser.parse_args()

  (x_train, y_train), (x_test, y_test) = cifar10.load_data()

  print("x_train:", x_train.shape) # (50000, 32, 32, 3)
  print("y_train:", y_train.shape) # (50000, 1)
  print("x_test:", x_test.shape)   # (10000, 32, 32, 3)
  print("y_test:", y_test.shape)   # (10000, 1)

  # Normalize for better performance in training.
  x_train = normalize(x_train)
  x_test = normalize(x_test)

  # Create one-hot encodings of the class labels 0..9.
  y_train = to_categorical(y_train, NUM_CLASSES) # shape = (50000, 10)
  y_test = to_categorical(y_test, NUM_CLASSES)   # shape = (10000, 10)

  model = build_model(args.arch)
  print(model.summary())

  train_model(model, x_train, y_train)

  model.evaluate(x_test, y_test)

  plot_results(model, x_test, y_test)


main()
