import tensorflow as tf
from .net import Net


class Softmax(Net):

  def __init__(self, **kwargs):
    self.output_dim = kwargs.get('output_dim', 1)
    return

  def inference(self, data):

    feature_dim = data.get_shape()[1].value

    with tf.name_scope('weights'):
      W = tf.Variable(tf.zeros([feature_dim, self.output_dim]))
    with tf.name_scope('biases'):
      b = tf.Variable(tf.zeros([self.output_dim]), name='bias')
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
