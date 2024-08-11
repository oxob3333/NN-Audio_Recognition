import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from tensorflow.keras import layers
from tensorflow.keras import models

class Nueva_prediccion:

  def __init__(self):
    dataset_dir = pathlib.Path('dataset')
    self.commands = np.array(tf.io.gfile.listdir(str(dataset_dir)))
    self.resultado =""

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
    waveform =  self.decode_audio(audio_binary)
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

  def get_spectrogram_and_label_id(self,audio, label):
    spectrogram =  self.get_spectrogram(audio)
    label_id = tf.argmax(label ==  self.commands)
    return spectrogram, label_id

  def preprocess_dataset(self,files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(
        map_func= self.get_waveform_and_label,
        num_parallel_calls=tf.data.AUTOTUNE)
    output_ds = output_ds.map(
        map_func= self.get_spectrogram_and_label_id,
        num_parallel_calls=tf.data.AUTOTUNE)
    return output_ds

  def iniciar(self):
    sample = pathlib.Path('prueba')

    

    new_model = tf.keras.models.load_model('modelo/modelo_audio.keras')

    sample_file = sample/'grabacion.wav'

    try:
      # Preprocesar el archivo de audio de muestra para identificar la palabra
      sample_ds =  self.preprocess_dataset([str(sample_file)])
    except:
      print("ERROR - archivo de audio no existe.")
      return 

    f3 = plt.figure()

    for spectrogram, label in sample_ds.batch(1):
      prediction = new_model(spectrogram)
      score = tf.nn.softmax(prediction[0])
      print("\nLa palabra probablemente sea: ",self.commands[np.argmax(score)])
      plt.figure(figsize=(12.80, 7.20))
      plt.bar( self.commands, score)
      plt.title('Grafica de Predicciones')
      plt.savefig("predict_muestra")
      self.resultado=str(self.commands[np.argmax(score)])
