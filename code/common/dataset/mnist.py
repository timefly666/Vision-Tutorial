import os
import numpy as np
from .dataset import Dataset
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def load_dataset(data_dir, split):
  print('downloading mnist dataset')
  mnist_data = input_data.read_data_sets(data_dir, one_hot=True)

  images = getattr(mnist_data, split).images.reshape([-1, 784])
  labels = getattr(mnist_data, split).labels.astype(np.float32)

  return images, labels


class MNIST(Dataset):

  def __init__(self, **kwargs):
    self.count = kwargs.get('count', None)
    self.buffer_size = kwargs.get('buffer_size', 10000)
    self.batch_size = kwargs.get('batch_size', 50)

    self.data_dir = kwargs.get('data_dir', None)
    self.split = kwargs.get('split', 'train')

    if self.split not in ['train', 'validation', 'test']:
      raise ValueError('unsupported dataset mode!')

    # download mnist data
    images, labels = load_dataset(self.data_dir, self.split)

    # build dataset
    dataset = tf.contrib.data.Dataset.from_tensor_slices((images, labels))
    if self.buffer_size is not None:
      dataset = dataset.shuffle(buffer_size=self.buffer_size)
    dataset = dataset.repeat(self.count)
    dataset = dataset.batch(self.batch_size)

    with tf.name_scope('input'):
      self._iterator = dataset.make_one_shot_iterator()
      self._batch = self._iterator.get_next()

      # image = tf.reshape(self._batch[0], [-1, 28, 28, 1])
      # tf.summary.image('image', image, 10)

  def batch(self):
    return self._batch

  def shape(self):
    return self._iterator.output_shapes
