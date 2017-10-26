from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from datetime import datetime

import tensorflow as tf
from .solver import Solver


class StandardSolver(Solver):

  def __init__(self, dataset, net, **kwargs):
    self.summary_iter = int(kwargs.get('summary_iter', 100))
    self.summary_dir = kwargs.get('summary_dir', 'cache/summary')
    self.snapshot_iter = int(kwargs.get('snapshot_iter', 1000))
    self.snapshot_dir = kwargs.get('snapshot_dir', 'cache/model')

    self.checkpoint = kwargs.get('checkpoint', None)
    self.summary_classes = kwargs.get('summary_classes', [])

    self.learning_rate = float(kwargs.get('learning_rate', 0.1))
    self.max_steps = int(kwargs.get('max_steps', 10000))
    self.optimizer = kwargs.get('optimizer', 'sgd')

    self.dataset = dataset
    self.net = net

    self.decay_steps = kwargs.get('decay_steps', self.max_steps // 3)
    self.decay_rate = kwargs.get('decay_rate', 0.9)

  def build_optimizer(self, loss):
    with tf.variable_scope('optimizer'):
      global_step = tf.contrib.framework.get_or_create_global_step()

      if self.optimizer == 'sgd':
        learning_rate = tf.train.exponential_decay(
            self.learning_rate,
            global_step,
            self.decay_steps,
            self.decay_rate,
            staircase=True)
        optimizer = tf.train.GradientDescentOptimizer
      elif self.optimizer == 'adam':
        learning_rate = self.learning_rate
        optimizer = tf.train.AdamOptimizer
      else:
        raise ValueError('unsupported optimizer!')

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = optimizer(learning_rate).minimize(
            loss, global_step=global_step)

    return train_op

  def build_train_net(self):
    data, labels = self.dataset.batch()

    self.layers = self.net.inference(data)
    self.loss = self.net.loss(self.layers, labels)
    self.total_loss = tf.add_n(tf.losses.get_regularization_losses() +
                               [self.loss])
    self.train_op = self.build_optimizer(self.total_loss)

    tf.summary.scalar('loss', self.total_loss)
    tf.summary.scalar('loss_without_regularization', loss)

  def build_eval_net(self):
    data, labels = self.dataset.batch()

    self.layers = self.net.inference(data)
    self.metrics = self.net.metric(self.layers, labels)
    self.summary_data = [labels, self.layers['predictions']]

    update_ops = []
    for key, value in self.metrics.iteritems():
      metric, update_op = value
      update_ops.append(update_op)
      tf.summary.scalar(key, metric)
      self.metrics[key] = metric
    self.update_op = tf.group(*update_ops)

  def build_inference_net(self):
    data, _ = self.dataset.batch()
    self.layers = self.net.inference(data)

  def train(self):
    self.build_train_net()
    init_op = [
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    ]

    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(
        os.path.join(self.summary_dir, 'train'))
    summary_writer.add_graph(tf.get_default_graph())

    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

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
          sec_per_batch = float(duration)
          format_str = ('%s: step %8d, loss = %.4f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print(format_str % (datetime.now(), step, loss, examples_per_sec,
                              sec_per_batch))

          sys.stdout.flush()

        if (step % self.snapshot_iter == 0) or (step == self.max_steps):
          saver.save(sess, self.snapshot_dir + '/model.ckpt', global_step=step)

  def eval(self):
    self.build_eval_net()
    saver = tf.train.Saver()
    init_op = [
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    ]
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(
        os.path.join(self.summary_dir, 'test'))
    summary_writer.add_graph(tf.get_default_graph())

    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
      sess.run(init_op)

      checkpoint = self.checkpoint
      if checkpoint is None:
        checkpoint = tf.train.latest_checkpoint(self.snapshot_dir)
        if not os.path.isfile(checkpoint + '.index'):
          print("====================")
          print("[error]: can't find checkpoint file: {}".format(checkpoint))
          sys.exit(0)

      print("load checkpoint file: {}".format(checkpoint))
      num_iter = int(checkpoint.split('-')[-1])

      saver.restore(sess, checkpoint)

      step = 0
      while True:
        try:
          step = step + 1
          if len(self.summary_classes) == 0:
            sess.run(self.update_op)
          else:
            _, summary = sess.run([self.update_op, self.summary_data])
            for summary_class in self.summary_classes:
              summary_class.accumulate(summary, step)

          if step % self.summary_iter == 0:
            print("step: {}".format(step))

        except tf.errors.OutOfRangeError:
          results = sess.run([summary_op] + self.metrics.values())
          summary = results[0]
          metrics = results[1:]
          for key, metric in zip(self.metrics.keys(), metrics):
            print("{}: {}".format(key, metric))

          summary_writer.add_summary(summary, num_iter)
          for summary_class in self.summary_classes:
            summary_class.summarize()

          break

  def predict(self):
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
