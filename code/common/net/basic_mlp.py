import tensorflow as tf
from .net import Net
from .layer.basic_layer import linear_relu, linear


class BasicMLP(Net):

  def __init__(self, **kwargs):
    self.output_size = kwargs.get('output_size', 1)
    self.mode = kwargs.get('output_size', 'train')
    return

  def inference(self, data):
    with tf.variable_scope('hidden1'):
      hidden1 = linear_relu(data, 128)

    with tf.variable_scope('hidden2'):
      hidden2 = linear_relu(hidden1, 32)

    with tf.variable_scope('softmax_linear'):
      y = linear(hidden2, self.output_size)

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
