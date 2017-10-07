from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

from common.dataset.mnist import MNIST as Dataset
from common.net.mlp import MLP as Net
from common.solver.basic_solver import BasicSolver


def main():
  args = parse_args()

  with tf.Graph().as_default():
    dataset = Dataset(data_dir=args.data_dir)
    net = Net(output_dim=10)
    solver = BasicSolver(dataset, net)
    solver.train()

  with tf.Graph().as_default():
    dataset = Dataset(data_dir=args.data_dir, split='test', count=1)
    net = Net(output_dim=10)
    solver = BasicSolver(dataset, net)
    solver.test()


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='../../dataset/mnist',
      help='directory for dataset')
  return parser.parse_args()


if __name__ == '__main__':
  main()
