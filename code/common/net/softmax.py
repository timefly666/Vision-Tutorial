import tensorflow as tf
from .net import Net


class Softmax(Net):

  def __init__(self, **kwargs):
    self.output_size = kwargs.get('output_size', 1)
    return

  def inference(self, data):

    feature_size = data.get_shape()[1].value

    with tf.name_scope('weights'):
      W = tf.Variable(tf.zeros([feature_size, self.output_size]))
    with tf.name_scope('biases'):
      b = tf.Variable(tf.zeros([self.output_size]), name='bias')
    with tf.name_scope('y'):
      y = tf.matmul(data, W) + b
    with tf.name_scope('probs'):
      probs = tf.nn.softmax(y)

    return {'logits': y, 'probs': probs}

  def loss(self, layers, labels):
    logits = layers['logits']

    with tf.variable_scope('loss'):
      loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return loss

  def metric(self, layers, labels):
    probs = layers['probs']
    with tf.variable_scope('metric'):
      metric, update_op = tf.metrics.accuracy(
          labels=tf.argmax(labels, 1), predictions=tf.argmax(probs, 1))
    return {'update': update_op, 'accuracy': metric}
