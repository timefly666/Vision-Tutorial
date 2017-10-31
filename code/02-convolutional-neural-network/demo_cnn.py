from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from common.dataset.tfrecord import TFRecord as Dataset
from common.solver.basic_solver import BasicSolver as Solver
from net.basic_cnn import BasicCNN as Net


def plot_sample(x, y):
  img = x.reshape(96, 96)
  plt.imshow(img, cmap='gray')
  plt.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)
  plt.show()


def parse_example(example_proto):
  features = {
      "data": tf.FixedLenFeature((9216), tf.float32),
      "label": tf.FixedLenFeature((30), tf.float32, default_value=[0.0] * 30),
  }
  parsed_features = tf.parse_single_example(example_proto, features)
  image = tf.reshape(parsed_features["data"], (96, 96, -1))
  return image, parsed_features["label"]


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='../../dataset/kaggle',
      help='directory for dataset')
  return parser.parse_args()


def main():
  args = parse_args()
  file_dict = {
      'train': os.path.join(args.data_dir, 'train.tfrecords'),
      'eval': os.path.join(args.data_dir, 'test.tfrecords')
  }

  with tf.Graph().as_default():
    dataset = Dataset(
        file_dict=file_dict,
        split='train',
        parse_function=parse_example,
        batch_size=50)
    net = Net(output_size=30)
    solver = Solver(dataset, net, max_steps=200, summary_iter=10)
    solver.train()

  with tf.Graph().as_default():
    dataset = Dataset(
        file_dict=file_dict,
        split='eval',
        parse_function=parse_example,
        batch_size=50)
    net = Net(output_size=30)
    solver = Solver(dataset, net)
    solver.eval()

  with tf.Graph().as_default():
    dataset = Dataset(
        file_dict=file_dict,
        split='eval',
        parse_function=parse_example,
        batch_size=1)
    net = Net(output_size=30)
    solver = Solver(dataset, net)
    results = solver.predict()
  plot_sample(results['data'], results['predictions'][0])


if __name__ == '__main__':
  main()
