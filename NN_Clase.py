import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from tensorflow.keras import layers
from tensorflow.keras import models


class NN_audio:

  def __init__(self,iteraciones):
    self.iteraciones = iteraciones
    self.AUTOTUNE = tf.data.AUTOTUNE
    self.DATASET_PATH = 'dataset'

    self.data_dir = pathlib.Path(self.DATASET_PATH)

    self.commands = np.array(tf.io.gfile.listdir(str(self.data_dir)))

  def decode_audio(self,audio_binary):
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    return tf.squeeze(audio, axis=-1)

  def get_label(self,file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    return parts[-2]

  def get_waveform_and_label(self,file_path):
    label = self.get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = self.decode_audio(audio_binary)
    return waveform, label

  def get_spectrogram(self,waveform):
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32)
    waveform = tf.cast(waveform, dtype=tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

  def plot_spectrogram(self,spectrogram, ax):
    if len(spectrogram.shape) > 2:
      assert len(spectrogram.shape) == 3
      spectrogram = np.squeeze(spectrogram, axis=-1)
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
    seed = 6
    tf.random.set_seed(seed)
    np.random.seed(seed)


    print('\nComandos: ', self.commands)

    filenames = tf.io.gfile.glob(str(self.data_dir) + '/*/*.wav')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)

    print('\nNumero total de muestras: ', num_samples)

    valores_de_entrada = int(float(num_samples))

    train_files = filenames[:valores_de_entrada]
    val_files = filenames[:valores_de_entrada]
    test_files = filenames[:valores_de_entrada]

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


    batch_size = int(len(filenames)*0.1)
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(self.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(self.AUTOTUNE)

    for spectrogram, _ in spectrogram_ds.take(1):
      input_shape = spectrogram.shape
    num_labels = len(self.commands)

    norm_layer = layers.Normalization()
    norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Resizing(50, 50),
        norm_layer,
        layers.Conv2D(8, 3, activation='relu'),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(75, activation='relu'),
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
    )

    test_audio = []
    test_labels = []

    for audio, label in test_ds:
      test_audio.append(audio.numpy())
      test_labels.append(label.numpy())

    f1 = plt.figure()


    metrics = history.history
    plt.plot(history.epoch, metrics['loss'],metrics['val_loss'])
    plt.legend(['Entrenamiento','Validacion'])
    plt.autoscale()
    plt.savefig("error")

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    model.save('modelo/modelo_audio.keras')

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