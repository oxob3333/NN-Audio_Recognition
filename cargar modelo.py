import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from main import preprocess_dataset

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

import speech_recognition as sr

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()

DATASET_PATH = 'mini_speech_commands'

data_dir = pathlib.Path(DATASET_PATH)
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
valid_train = int(float(num_samples)*0.1)
test_files = filenames[-valid_train:]
test_ds = preprocess_dataset(test_files)

test_audio = []
test_labels = []

for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

loss, acc = new_model.evaluate(test_audio, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))