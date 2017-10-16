from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from datetime import datetime

import tensorflow as tf
from .solver import Solver


class BasicSolver(Solver):

  def __init__(self, dataset, net, **kwargs):
    self.summary_iter = int(kwargs.get('summary_iter', 100))
    self.summary_dir = kwargs.get('summary_dir', 'summary')
    self.snapshot_iter = int(kwargs.get('snapshot_iter', 100000))
    self.snapshot_dir = kwargs.get('snapshot_dir', 'cache')

    self.learning_rate = float(kwargs.get('learning_rate', 0.1))
    self.max_steps = int(kwargs.get('max_steps', 4000))

    self.dataset = dataset
    self.net = net

  def build_optimizer(self):
    with tf.variable_scope('optimizer'):
      train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
          self.total_loss)
    return train_op

  def build_train_net(self):
    data, labels = self.dataset.batch()

    self.layers = self.net.inference(data)
    self.loss = self.net.loss(self.layers, labels)
    self.total_loss = tf.add_n(
        tf.get_collection('losses') + [self.loss], name='total_loss')
    self.train_op = self.build_optimizer()

    for loss_layer in tf.get_collection('losses') + [self.loss]:
      tf.summary.scalar(loss_layer.op.name, loss_layer)

  def build_test_net(self):
    data, labels = self.dataset.batch()

    self.layers = self.net.inference(data)
    self.metrics = self.net.metric(self.layers, labels)
    self.update_op = self.metrics.pop('update')

    for key, value in self.metrics.iteritems():
      tf.summary.scalar(key, value)

  def build_inference_net(self):
    data, _ = self.dataset.batch()
    self.layers = self.net.inference(data)

  def train(self):
    self.build_train_net()
    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(
        os.path.join(self.summary_dir, 'train'))
    summary_writer.add_graph(tf.get_default_graph())

    config = tf.ConfigProto(log_device_placement=False)
    with tf.Session(config=config) as sess:
      sess.run(init_op)

      for step in xrange(1, self.max_steps + 1):
        start_time = time.time()
        sess.run(self.train_op)
        duration = time.time() - start_time

        if step % self.summary_iter == 0:
          summary, loss = sess.run([summary_op, self.loss])
          summary_writer.add_summary(summary, step)

          examples_per_sec = self.dataset.batch_size / duration
          format_str = ('step %6d: loss = %.4f (%.1f examples/sec)')
          print(format_str % (step, loss, examples_per_sec))

          sys.stdout.flush()

        if (step % self.snapshot_iter == 0) or (step == self.max_steps):
          saver.save(sess, self.snapshot_dir + '/model.ckpt', global_step=step)

  def test(self):
    self.build_test_net()
    saver = tf.train.Saver()
    init_op = [
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    ]
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(
        os.path.join(self.summary_dir, 'test'))
    summary_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
      sess.run(init_op)

      checkpoint = tf.train.latest_checkpoint(self.snapshot_dir)
      if not os.path.isfile(checkpoint + '.index'):
        print("====================")
        print("[error]: can't find checkpoint file: {}".format(checkpoint))
        sys.exit(0)

      print("load checkpoint file: {}".format(checkpoint))
      num_iter = int(checkpoint.split('-')[-1])

      saver.restore(sess, checkpoint)

      while True:
        try:
          sess.run(self.update_op)
        except tf.errors.OutOfRangeError:
          results = sess.run([summary_op] + self.metrics.values())
          summary = results[0]
          metrics = results[1:]
          for key, metric in zip(self.metrics.keys(), metrics):
            print("{}: {}".format(key, metric))
          summary_writer.add_summary(summary, num_iter)
          break

  def inference(self):
    self.build_inference_net()
    saver = tf.train.Saver()
    init_op = [
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    ]
    with tf.Session() as sess:
      sess.run(init_op)

      checkpoint = tf.train.latest_checkpoint(self.snapshot_dir)
      if not os.path.isfile(checkpoint + '.index'):
        print("====================")
        print("[error]: can't find checkpoint file: {}".format(checkpoint))
        sys.exit(0)

      print("load checkpoint file: {}".format(checkpoint))
      saver.restore(sess, checkpoint)

      layers = sess.run(self.layers)
      return layers
