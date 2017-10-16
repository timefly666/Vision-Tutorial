import os
import numpy as np
from .dataset import Dataset
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class TFRecord(Dataset):

  def __init__(self, **kwargs):
    self.files = kwargs.get('files')
    self.parse_function = kwargs.get('parse_function')

    self.count = kwargs.get('count', None)
    self.buffer_size = kwargs.get('buffer_size', 7050 * 1001)
    self.batch_size = kwargs.get('batch_size', 1)

    dataset = tf.contrib.data.TFRecordDataset(self.files)

    dataset = dataset.map(self.parse_function)
    if self.buffer_size is not None:
      dataset = dataset.shuffle(buffer_size=self.buffer_size)
    dataset = dataset.repeat(self.count)
    dataset = dataset.batch(self.batch_size)

    with tf.name_scope('input'):
      self._iterator = dataset.make_one_shot_iterator()
      self._batch = self._iterator.get_next()

      # tf.summary.image('image', self._batch[0], 10)

  def batch(self):
    return self._batch

  def shape(self):
    return self._iterator.output_shapes
