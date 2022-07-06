import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display


class NN_audio:

  def __init__(self,iteraciones):
    self.iteraciones = iteraciones
    self.AUTOTUNE = tf.data.AUTOTUNE
    self.DATASET_PATH = 'dataset'

    self.data_dir = pathlib.Path(self.DATASET_PATH)

    self.commands = np.array(tf.io.gfile.listdir(str(self.data_dir)))

  def decode_audio(self,audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(audio, axis=-1)

  def get_label(self,file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]

  def get_waveform_and_label(self,file_path):
    label = self.get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = self.decode_audio(audio_binary)
    return waveform, label

  def get_spectrogram(self,waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

  def plot_spectrogram(self,spectrogram, ax):
    if len(spectrogram.shape) > 2:
      assert len(spectrogram.shape) == 3
      spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

  def get_spectrogram_and_label_id(self,audio, label):
    spectrogram = self.get_spectrogram(audio)
    label_id = tf.argmax(label == self.commands)
    return spectrogram, label_id

  def preprocess_dataset(self,files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(
        map_func=self.get_waveform_and_label,
        num_parallel_calls=self.AUTOTUNE)
    output_ds = output_ds.map(
        map_func=self.get_spectrogram_and_label_id,
        num_parallel_calls=self.AUTOTUNE)
    return output_ds


  def iniciar(self):
# Set the seed value for experiment reproducibility.
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)


    print('\nComandos: ', self.commands)

    time.sleep(1)

    filenames = tf.io.gfile.glob(str(self.data_dir) + '/*/*.wav')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    print('\nNumber of total examples:', num_samples)

    valores_de_entrada = int(float(num_samples)*0.8)

    valid_train = int(float(num_samples)*0.2)

    train_files = filenames[:valores_de_entrada]
    val_files = filenames[valores_de_entrada:valores_de_entrada + valid_train]
    test_files = filenames

    print('Training set size: ', len(train_files))
    print('Validation set size: ', len(val_files))
    print('Test set size: ', len(test_files))



    files_ds = tf.data.Dataset.from_tensor_slices(train_files)

    waveform_ds = files_ds.map(
        map_func=self.get_waveform_and_label,
        num_parallel_calls=self.AUTOTUNE)

    spectrogram_ds = waveform_ds.map(
      map_func=self.get_spectrogram_and_label_id,
      num_parallel_calls=self.AUTOTUNE)

    train_ds = spectrogram_ds
    val_ds = self.preprocess_dataset(val_files)
    test_ds = self.preprocess_dataset(test_files)


    batch_size = int(len(filenames)*0.4)
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(self.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(self.AUTOTUNE)

    for spectrogram, _ in spectrogram_ds.take(1):
      input_shape = spectrogram.shape
    print('Input shape:', input_shape)
    num_labels = len(self.commands)

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(100, 100),
        # Normalize.
        norm_layer,
        layers.Conv2D(16, 3, activation='relu'),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(50, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=self.iteraciones,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

    test_audio = []
    test_labels = []

    for audio, label in test_ds:
      test_audio.append(audio.numpy())
      test_labels.append(label.numpy())

    f1 = plt.figure()


    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.savefig("error")

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    model.save('modelo/modelo_audio.h5')

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'\nPrecisión de la prueba: {test_acc:.0%}')

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    f2 = plt.figure()
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=self.commands,
                yticklabels=self.commands,
                annot=True, fmt='g')
                
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.savefig("confusion_matrix")

    f3 = plt.figure()


    sample_file = self.data_dir/'Adelante/grabacion.wav'

    sample_ds = self.preprocess_dataset([str(sample_file)])

    for spectrogram, label in sample_ds.batch(1):
      prediction = model(spectrogram)
      score = tf.nn.softmax(prediction[0])
      print("\nLa palabra probablemente sea: ",self.commands[np.argmax(score)])
      plt.bar(self.commands, score)
      plt.title(f'Predicciones para "{self.commands[label[0]]}"')
      plt.savefig("predict")
