from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def save_as_tfrecord(data_path, data, labels, **kargs):
  if not data_path.endswith('.tfrecords'):
    data_path = "{}.tfrecords".format(data_path)
  print("==================== save data [tfrecord]: {}".format(data_path))

  tfrecord_writer = tf.python_io.TFRecordWriter(data_path)
  for i in range(data.shape[0]):
    if i % 100 == 0:
      print("====={}/{}".format(i, data.shape[0]))
    feature = {}
    if labels is not None:
      feature['label'] = _float_list_feature(labels[i])
    feature['data'] = _float_list_feature(np.ravel(data[i]))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    tfrecord_writer.write(example.SerializeToString())


def load_dataset(data_dir, split='train', cols=None):
  """Load the dataset.

  Args:
    split: train/test
    cols: optional, defaults to `None`. A list of columns you're interested in.
      If specified only returns these columns.
  """
  print('load dataset: {}'.format(split))

  if split not in ['train', 'test']:
    raise ValueError('unsupported dataset split!')

  if split == 'train':
    df = pd.read_csv(os.path.join(data_dir, 'training.csv'))
  else:
    df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

  df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

  if cols:  # get a subset of columns
    df = df[list(cols) + ['Image']]

  print(df.count())  # prints the number of values for each column
  df = df.dropna()  # drop all rows that have missing values in them

  images = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
  images = images.astype(np.float32)

  if split == 'train':  # only train split has label target columns
    label = df[df.columns[:-1]].values
    label = (label - 48) / 48  # scale target coordinates to [-1, 1]
    label = label.astype(np.float32)
  else:
    label = None

  image_size = 96
  num_channels = 1
  images = images.reshape(-1, image_size, image_size, num_channels)

  return images, label


# converter the dataset to tfrecord format
def convert_to_tfrecord(dataset_dir, tfrecord_dir):
  data, labels = load_dataset(dataset_dir, 'train')
  save_as_tfrecord(os.path.join(tfrecord_dir, 'train'), data, labels)

  data, labels = load_dataset(dataset_dir, 'test')
  save_as_tfrecord(os.path.join(tfrecord_dir, 'test'), data, labels)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset_dir',
      type=str,
      default='../../dataset/kaggle',
      help='directory for dataset')
  parser.add_argument(
      '--tfrecord_dir',
      type=str,
      default='../../dataset/kaggle',
      help='directory for tfrecord')
  return parser.parse_args()


def main():
  args = parse_args()
  convert_to_tfrecord(args.dataset_dir, args.tfrecord_dir)


if __name__ == '__main__':
  main()
