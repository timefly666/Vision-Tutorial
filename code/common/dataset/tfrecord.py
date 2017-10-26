import os
import numpy as np

import tensorflow as tf
from .dataset import Dataset


class TFRecord(Dataset):

  def __init__(self, **kwargs):
    self.count = kwargs.get('count', None)
    self.buffer_size = kwargs.get('buffer_size', 10000)
    self.batch_size = kwargs.get('batch_size', 50)

    self.file_dict = kwargs.get('file_dict')
    self.parse_function = kwargs.get('parse_function')

    self.split = kwargs.get('split', 'train')
    self.splits = self.file_dict.keys()
    self._datasets = {}

    for split, files in self.file_dict.items():
      dataset = tf.contrib.data.TFRecordDataset(files)
      dataset = dataset.map(self.parse_function)

      if self.buffer_size is not None:
        dataset = dataset.shuffle(buffer_size=self.buffer_size)

      if split == 'train':
        dataset = dataset.repeat(self.count)

      dataset = dataset.batch(self.batch_size)
      self._datasets[split] = dataset

  def batch(self, split=None):
    if split is None:
      split = self.split
    if split not in self.splits:
      raise ValueError('unsupported dataset split!')

    dataset = self._datasets[split]
    with tf.name_scope('input'):
      iterator = dataset.make_one_shot_iterator()
      batch = iterator.get_next()
      # tf.summary.image('image', batch[0], 10)
    return batch

  def shape(self, split=None):
    if split is None:
      split = self.split
    if split not in self.splits:
      raise ValueError('unsupported dataset split!')

    dataset = self._datasets[split]
    return dataset.output_shapes
